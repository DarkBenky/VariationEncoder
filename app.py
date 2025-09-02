import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from main import create_feature_vector_from_descriptions, buildEncoder, buildDecoder, buildClassifier, buildInverseClassifier, generate_image_from_features, get_celeba_attribute_names

@st.cache_resource
def load_models():
    latent_dim = 256  # Changed from 128 to 256 to match saved weights
    
    encoder = buildEncoder(latent_dim)
    decoder = buildDecoder(latent_dim)
    classifier = buildClassifier(latent_dim)
    inverse_classifier = buildInverseClassifier(latent_dim)
    
    try:
        encoder.load_weights('encoder.weights.h5')
        decoder.load_weights('decoder.weights.h5')
        classifier.load_weights('classifier.weights.h5')
        inverse_classifier.load_weights('inverse_classifier.weights.h5')
        return encoder, decoder, classifier, inverse_classifier
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None, None, None

st.set_page_config(page_title="Face Generation", page_icon="🎭", layout="wide")
st.title("🎭 Face Generation with VAE")
st.markdown("Generate faces by adjusting facial attributes")

encoder, decoder, classifier, inverse_classifier = load_models()

if inverse_classifier is None or decoder is None:
    st.error("Models not loaded. Please ensure model weights exist.")
    st.stop()

feature_names = get_celeba_attribute_names()

st.sidebar.header("Facial Attributes")
st.sidebar.markdown("Adjust sliders to control facial features (0=absent, 1=strong)")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Controls")
    
    # Add image upload section
    st.markdown("**Upload Image to Extract Features:**")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        if st.button("Extract Features", type="secondary"):
            with st.spinner("Extracting features..."):
                try:
                    # Load and preprocess image
                    image = Image.open(uploaded_file)
                    image = image.convert('RGB')
                    image = image.resize((256, 256))
                    image_array = np.array(image) / 255.0
                    image_tensor = tf.expand_dims(image_array, 0)
                    
                    # Get latent representation and predict attributes
                    latent_code = encoder(image_tensor, training=False)
                    predicted_attributes = classifier(latent_code, training=False)
                    
                    # Update sliders with predicted values
                    for i, feature in enumerate(feature_names):
                        key = f"slider_{feature}"
                        st.session_state[key] = float(predicted_attributes[0][i].numpy())
                    
                    st.success("Features extracted! Check the sliders.")
                    
                    # Show uploaded image
                    st.image(image, caption="Uploaded Image", width=200)
                    
                except Exception as e:
                    st.error(f"Error extracting features: {e}")
    
    if st.button("Reset All", type="secondary"):
        for key in st.session_state.keys():
            if key.startswith("slider_"):
                st.session_state[key] = 0.0
    
    st.markdown("**Quick Presets:**")
    if st.button("Young Woman"):
        st.session_state.update({
            "slider_Young": 0.8,
            "slider_Attractive": 0.7,
            "slider_Smiling": 0.6,
            "slider_No_Beard": 1.0,
            "slider_Wearing_Lipstick": 0.5
        })
    
    if st.button("Older Man"):
        st.session_state.update({
            "slider_Male": 0.9,
            "slider_Young": 0.1,
            "slider_No_Beard": 0.3,
            "slider_Receding_Hairline": 0.6
        })

feature_values = []
organized_features = {
    "Physical Traits": ["Young", "Male", "Attractive", "Chubby", "Pale_Skin"],
    "Hair": ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Bald", "Receding_Hairline", "Bangs", "Straight_Hair", "Wavy_Hair"],
    "Facial Hair": ["5_o_Clock_Shadow", "Goatee", "Mustache", "No_Beard", "Sideburns"],
    "Eyes & Eyebrows": ["Arched_Eyebrows", "Bushy_Eyebrows", "Bags_Under_Eyes", "Narrow_Eyes", "Eyeglasses"],
    "Facial Features": ["Big_Lips", "Big_Nose", "High_Cheekbones", "Pointy_Nose", "Double_Chin", "Oval_Face", "Rosy_Cheeks"],
    "Expression": ["Smiling", "Mouth_Slightly_Open"],
    "Makeup & Accessories": ["Heavy_Makeup", "Wearing_Lipstick", "Wearing_Earrings", "Wearing_Hat", "Wearing_Necklace", "Wearing_Necktie"],
    "Other": ["Blurry"]
}

with st.sidebar:
    for category, features in organized_features.items():
        with st.expander(category):
            for feature in features:
                key = f"slider_{feature}"
                value = st.slider(
                    feature.replace('_', ' '), 
                    0.0, 1.0, 0.0, 0.1,
                    key=key
                )
                
                idx = feature_names.index(feature)
                while len(feature_values) <= idx:
                    feature_values.append(0.0)
                feature_values[idx] = value

while len(feature_values) < len(feature_names):
    feature_values.append(0.0)

with col2:
    st.subheader("Generated Face")
    
    if st.button("Generate Image", type="primary"):
        with st.spinner("Generating face..."):
            try:
                generated_image = generate_image_from_features(
                    feature_values, inverse_classifier, decoder
                )
                
                generated_image = np.clip(generated_image, 0, 1)
                
                st.image(generated_image, caption="Generated Face", width=400)
                
                active_features = []
                for i, (feature, value) in enumerate(zip(feature_names, feature_values)):
                    if value > 0.3:
                        active_features.append(f"{feature.replace('_', ' ')}: {value:.1f}")
                
                if active_features:
                    st.write("**Active Features:**")
                    st.write(", ".join(active_features[:10]))
                    if len(active_features) > 10:
                        st.write(f"...and {len(active_features) - 10} more")
                
            except Exception as e:
                st.error(f"Error generating image: {e}")

st.markdown("---")
st.markdown("**Tips:**")
st.markdown("- Start with basic features like 'Young', 'Male', or 'Attractive'")
st.markdown("- Try different combinations for unique faces")
st.markdown("- Values between 0.3-0.8 often work best")
