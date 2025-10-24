
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
from PIL import Image
import pickle
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, concatenate

# --- Fonctions utilitaires ---


def add_noise_to_image(image_array, noise_factor):
    """Adds Gaussian noise to an image array (normalized 0-1) with a reduced scale."""
    # Reduced scale from 1.0 to 0.5 to make the noise less extreme and more realistic.
    noisy_image = image_array + noise_factor * \
        np.random.normal(loc=0.0, scale=0.5, size=image_array.shape)
    noisy_image = np.clip(noisy_image, 0., 1.)
    return noisy_image


def augment_and_add_noise(images_tensor, noise_range=(0.05, 0.1)):
    """Applique une augmentation (flip) et ajoute un bruit gaussien variable."""
    clean_augmented_images = []
    noisy_images = []

    for img in images_tensor:
        # 1. Augmentation (50% de chance d'un flip horizontal)
        augmented_img = img
        if tf.random.uniform(()) > 0.5:
            augmented_img = tf.image.flip_left_right(augmented_img)

        clean_augmented_images.append(augmented_img.numpy())

        # 2. Ajout de bruit variable
        noise_factor = tf.random.uniform(
            (), minval=noise_range[0], maxval=noise_range[1])
        noise = tf.random.normal(shape=tf.shape(augmented_img))
        noisy_img_tensor = augmented_img + noise_factor * noise
        noisy_img_tensor = tf.clip_by_value(noisy_img_tensor, 0., 1.)
        noisy_images.append(noisy_img_tensor.numpy())

    return np.array(noisy_images), np.array(clean_augmented_images)


# --- Configuration de la page ---
st.set_page_config(page_title="Pipeline de Traitement d'Images", layout="wide")

# --- Fonctions de chargement des modèles (avec mise en cache) ---

# Métrique SSIM personnalisée pour le modèle de débruitage


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


@st.cache_resource
def load_binary_classifier():
    model_path = 'livrable 1/binary transfert learning/binary_classifier_model_finetuned.h5'
    model = load_model(model_path)
    return model


@st.cache_resource
def load_denoising_model():
    model_path = 'livrable2/best_model_unet.keras'
    custom_objects = {'ssim_metric': ssim_metric}
    model = load_model(model_path, custom_objects=custom_objects)
    return model


@st.cache_resource
def load_tokenizer():
    tokenizer_path = 'livrable3/tokenizer_best_complet.pickle'
    if not os.path.exists(tokenizer_path):
        st.error(
            f"Tokenizer file not found at {tokenizer_path}. Captioning will not work.")
        return None
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# --- Définition des Modèles de Captioning Personnalisés (Corrigé pour le chargement) ---


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features (batch_size, 64, 256)

        # S'assurer que hidden est de rang 2 avant l'expansion
        if len(hidden.shape) == 1:
            hidden = tf.expand_dims(hidden, 0)

        hidden_with_time_axis = tf.expand_dims(
            hidden, 1)  # (batch_size, 1, units)

        # score shape == (batch_size, 64, 1)
        score = self.V(tf.nn.tanh(self.W1(features) +
                       self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after reduce_sum == (batch_size, EMBEDDING_DIM)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # --- NOUVEAU : Couche Dropout pour l'embedding (taux de 30-50%) ---
        self.dropout_embed = tf.keras.layers.Dropout(0.4)

        self.gru_cell = tf.keras.layers.GRUCell(
            self.units,
            recurrent_initializer='glorot_uniform'
        )

        self.fc1 = tf.keras.layers.Dense(self.units)

        # --- NOUVEAU : Couche Dropout pour la sortie (taux de 30-50%) ---
        self.dropout_fc = tf.keras.layers.Dropout(0.4)

        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    # --- MODIFIÉ : Ajout de l'argument 'training' ---
    def call(self, x, features, hidden, training=False):
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)

        # --- NOUVEAU : Appliquer le dropout (seulement pendant l'entraînement) ---
        x = self.dropout_embed(x, training=training)

        x = tf.squeeze(x, axis=1)
        x_combined = tf.concat([context_vector, x], axis=-1)
        output_gru, [state] = self.gru_cell(x_combined, states=[hidden])

        x = self.fc1(output_gru)

        # --- NOUVEAU : Appliquer le dropout (seulement pendant l'entraînement) ---
        x = self.dropout_fc(x, training=training)

        x = tf.nn.relu(x)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# --- Fonctions de chargement des modèles ---


@st.cache_resource
def build_and_load_captioning_models():
    tokenizer = load_tokenizer()
    if tokenizer is None:
        return None, None, None, None

    # --- Recréer l'architecture et les instances de modèle ---
    embedding_dim = 256
    units = 768
    # Correction: Utiliser la taille exacte du tokenizer
    vocab_size = len(tokenizer.word_index)

    # --- Instantiate models ---
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    # --- Créer le modèle d'extraction de caractéristiques ---
    image_model = InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # --- Charger les poids ---
    encoder_weights_path = 'livrable3/encoder_complet.h5'
    decoder_weights_path = 'livrable3/decoder_complet.h5'

    if not all([os.path.exists(p) for p in [encoder_weights_path, decoder_weights_path]]):
        st.error("Fichiers de poids (_best_weights.h5) non trouvés.")
        return None, None, None, None

    try:
        # Étape cruciale : "Construire" les modèles avec un appel factice
        # pour que Keras puisse allouer les poids avant de les charger.
        img_tensor_val = tf.random.uniform(
            (1, 64, 2048))  # Shape de sortie d'InceptionV3
        features = encoder(img_tensor_val)
        hidden = decoder.reset_state(batch_size=1)
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        decoder(dec_input, features, hidden)

        # Maintenant, charger les poids
        encoder.load_weights(encoder_weights_path)
        decoder.load_weights(decoder_weights_path)
        st.success("Poids des modèles de captioning chargés avec succès.")
        return image_features_extract_model, encoder, decoder, tokenizer
    except Exception as e:
        st.error(f"Erreur lors du chargement des poids des modèles : {e}")
        return None, None, None, None


def generate_caption(image_array_normalized, image_features_extract_model, encoder, decoder, tokenizer, max_length=50):
    if any(p is None for p in [image_features_extract_model, encoder, decoder, tokenizer]):
        return "Modèles de captioning non chargés."

    hidden = decoder.reset_state(batch_size=1)

    # Pré-traitement de l'image pour InceptionV3
    img_pil = Image.fromarray((image_array_normalized * 255).astype(np.uint8))
    img_resized = np.array(img_pil.resize((299, 299)))
    img_preprocessed = tf.keras.applications.inception_v3.preprocess_input(
        img_resized)
    temp_input = tf.expand_dims(img_preprocessed, 0)

    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    # Lignes 244-254 à remplacer par ceci :

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    beam_width = 3  # Utiliser la même largeur de faisceau que le notebook
    max_length = 50  # S'assurer que max_length est défini

    # Le 'faisceau' (beam) stocke des tuples : (séquence, probabilité_log, état_caché)
    live_beams = []

    # --- Premier passage (initialisation) ---
    predictions, hidden, _ = decoder(
        dec_input, features, hidden, training=False)

    log_probs = tf.nn.log_softmax(predictions[0])
    top_k_probs, top_k_indices = tf.nn.top_k(log_probs, k=beam_width)

    # Créer les premiers faisceaux
    for i in range(beam_width):
        live_beams.append(
            ([tokenizer.word_index['<start>'], top_k_indices[i].numpy()],  # Séquence
             top_k_probs[i].numpy(),                 # Score (log-prob)
             hidden)                                 # État caché
        )

    # --- Boucle de génération (Beam Search) ---
    completed_beams = []

    for _ in range(max_length - 1):
        if not live_beams:
            break

        new_live_beams = []
        all_candidates = []

        for seq, score, h_state in live_beams:
            # L'entrée est le DERNIER mot de cette séquence
            dec_input = tf.expand_dims([seq[-1]], 0)

            predictions, new_h_state, _ = decoder(
                dec_input, features, h_state, training=False)

            log_probs = tf.nn.log_softmax(predictions[0])
            top_k_probs, top_k_indices = tf.nn.top_k(log_probs, k=beam_width)

            # Ajouter les nouveaux candidats
            for i in range(beam_width):
                new_seq = seq + [top_k_indices[i].numpy()]
                new_score = score + top_k_probs[i].numpy()
                all_candidates.append((new_seq, new_score, new_h_state))

        # Trier tous les candidats et garder les 'beam_width' meilleurs
        sorted_candidates = sorted(
            all_candidates, key=lambda x: x[1], reverse=True)

        # Réduire la largeur du faisceau si certains sont terminés
        beam_width = len(live_beams)
        top_candidates = sorted_candidates[:beam_width]

        # Séparer les faisceaux 'terminés' des 'vivants'
        for cand_seq, cand_score, cand_h_state in top_candidates:
            if cand_seq[-1] == tokenizer.word_index['<end>']:
                completed_beams.append((cand_seq, cand_score / len(cand_seq)))
            else:
                new_live_beams.append((cand_seq, cand_score, cand_h_state))

        live_beams = new_live_beams

    # --- Finalisation ---
    if not completed_beams:
        completed_beams = [(s, sc / len(s)) for s, sc, h in live_beams]

    completed_beams.sort(key=lambda x: x[1], reverse=True)
    best_seq_tokens = completed_beams[0][0]

    result = [tokenizer.index_word.get(t, '<unk>') for t in best_seq_tokens]
    result = [w for w in result if w not in ('<start>', '<end>', '<pad>')]

    return ' '.join(result)


# --- Interface Streamlit ---
st.set_page_config(page_title="Pipeline de Traitement d'Images", layout="wide")

st.title("🖼️ Pipeline de Traitement d'Images")
st.write("Projet de démonstration pour l'entreprise TouNum. Ce pipeline analyse une image, la classifie, la nettoie et génère une légende.")

# Colonnes pour l'interface
col1, col2 = st.columns(2)

with col1:
    st.header("1. Téléversement de l'image")
    uploaded_file = st.file_uploader(
        "Choisissez une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Image téléversée", use_column_width=True)

# --- Logique du pipeline ---

if uploaded_file is not None:
    # Initialiser l'état de session pour suivre l'exécution de l'analyse
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
        st.session_state.is_photo = False
        st.session_state.classification_result = ""
        st.session_state.confidence = 0.0

    with col2:
        st.header("2. Analyse de l'image")

        # Le bouton ne fait que lancer l'analyse initiale et met à jour l'état
        if st.button("Lancer l'analyse"):
            with st.spinner("Étape 1/3 : Classification de l'image..."):
                classifier = load_binary_classifier()

                img_for_classification = image.resize((224, 224))
                img_array = np.array(img_for_classification) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)

                prediction = classifier.predict(img_batch)[0][0]

                # Sauvegarder les résultats dans l'état de session
                st.session_state.is_photo = prediction > 0.5
                st.session_state.classification_result = "Photo" if st.session_state.is_photo else "Autre"
                st.session_state.confidence = prediction if st.session_state.is_photo else 1 - prediction
                st.session_state.analysis_run = True

        # Ce bloc s'exécute à chaque rafraîchissement si l'analyse a été lancée une fois
        if st.session_state.analysis_run:

            # Créer les onglets pour organiser les résultats
            tab1, tab2, tab3 = st.tabs(
                ["📊 Classification", "✨ Débruitage", "✍️ Génération de Légende"])

            with tab1:
                st.subheader("Résultat de la Classification")
                col1_metric, col2_metric = st.columns(2)
                col1_metric.metric("Type d'image détecté",
                                   st.session_state.classification_result)
                col2_metric.metric("Niveau de Confiance",
                                   f"{st.session_state.confidence:.2%}")

                if not st.session_state.is_photo:
                    st.warning(
                        "L'image n'est pas une photo. Les étapes de débruitage et de génération de légende sont désactivées.")

            if st.session_state.is_photo:
                with tab2:
                    st.subheader("Démonstration du Débruitage")

                    noise_factor = st.slider("Intensité du bruit gaussien", min_value=0.0,
                                             max_value=0.5, value=0.1, step=0.01, key="noise_slider_main")

                    img_for_denoising_original = image.resize((128, 128))
                    img_array_denoise_original = np.array(
                        img_for_denoising_original) / 255.0

                    noisy_img_array = add_noise_to_image(
                        img_array_denoise_original, noise_factor)

                    with st.spinner("Application du modèle de débruitage..."):
                        denoiser = load_denoising_model()
                        noisy_img_batch = np.expand_dims(
                            noisy_img_array, axis=0)
                        denoised_batch = denoiser.predict(noisy_img_batch)
                        denoised_image = np.clip(denoised_batch[0], 0.0, 1.0)

                    col1_denoise, col2_denoise = st.columns(2)
                    with col1_denoise:
                        st.image(
                            noisy_img_array, caption="Image Bruitée (pour démo)", use_column_width=True)
                    with col2_denoise:
                        st.image(
                            denoised_image, caption="Image Débruitée (Résultat)", use_column_width=True)

                # --- Étape 3: Captioning ---
                st.subheader("Génération de légende (Captioning)")
                with st.spinner("Étape 3/3 : Génération de la légende..."):
                    image_features_extract_model, encoder, decoder, tokenizer = build_and_load_captioning_models()
                    if all(p is not None for p in [image_features_extract_model, encoder, decoder, tokenizer]):
                        # Utiliser l'image originale (redimensionnée et normalisée) pour le captioning pour plus de précision
                        image_for_captioning = np.array(image) / 255.0

                        caption = generate_caption(
                            image_for_captioning, image_features_extract_model, encoder, decoder, tokenizer)
                        st.success(f"**Légende générée :** {caption}")
                    else:
                        st.error(
                            "Impossible de charger les modèles de captioning ou le tokenizer.")
