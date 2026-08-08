import streamlit as st
import os

# 0 = all messages, 1 = hide INFO, 2 = hide INFO and WARNINGS, 3 = hide all
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- 1. SETUP PAGE & PATHS ---
st.set_page_config(page_title="Cat vs Dog AI", page_icon="🐶", layout="centered")

st.title("🐱 Cat vs. Dog CNN Classifier 🐶")
st.write("Upload a picture of a cat or a dog, and our Deep Learning AI will guess which one it is using Transfer Learning!")

# Find the exact folder this script is living in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to our new WEIGHTS file!
WEIGHTS_PATH = os.path.join(BASE_DIR, 'cat_dog_classifier.keras')

@st.cache_resource
def load_my_model():
    # --- THE BULLETPROOF METHOD ---
    # Because Google Colab and local computers often have different TensorFlow versions,
    # trying to load an entire model can cause crashes. 
    # Instead, we rebuild the "Blueprint" here, and just load the "Memories" (weights)!
    
    IMG_SIZE = 160 
    
    # 1. Rebuild the MobileNetV2 Base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None # We don't need Google's weights, we have our own!
    )
    base_model.trainable = False

    # 2. Rebuild our Custom Layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 3. Pour the saved memories (weights) into our blueprint!
    model.load_weights(WEIGHTS_PATH)
    
    return model

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model weights not found! The app is looking exactly here: {WEIGHTS_PATH}")
    st.error(f"THE REAL ERROR: {str(e)}")
    st.stop()


# --- 2. IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using the PIL library
    image = Image.open(uploaded_file)
    
    # Display the image on the website
    st.image(image, caption="Your Uploaded Image", use_column_width=True)
    st.markdown("---")
    
    # --- 3. PREPROCESS THE UPLOADED IMAGE ---
    with st.spinner("AI is analyzing the image..."):
        # 1. Resize to 160x160 (The exact size MobileNetV2 expects)
        img_resized = ImageOps.fit(image, (160, 160))
        
        # 2. Convert the image into a mathematical NumPy array
        img_array = np.asarray(img_resized)
        
        # Ensure it has 3 color channels (RGB). If someone uploads a black & white photo, this catches it!
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            st.error("Please upload a standard color image (RGB).")
            st.stop()
            
        # 3. Normalize the pixels! (Just like we did in training: -1 to 1)
        img_array = (img_array / 127.5) - 1.0
        
        # 4. Add a "batch" dimension. 
        # The model expects a list of images, so we put our 1 image inside a list
        img_batch = np.expand_dims(img_array, axis=0) # Shape becomes (1, 160, 160, 3)

        # --- 4. THE PREDICTION ---
        # The model outputs a probability between 0 and 1
        prediction = model.predict(img_batch)[0][0]
        
        # Decide the winner
        if prediction < 0.5:
            animal = "CAT 🐱"
            confidence = (1 - prediction) * 100
        else:
            animal = "DOG 🐶"
            confidence = prediction * 100
            
        # Display the result!
        st.success(f"The AI is **{confidence:.1f}%** confident this is a **{animal}**!")