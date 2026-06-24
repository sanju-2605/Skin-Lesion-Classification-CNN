import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("skin_lesion_model1.h5")

# Class labels
classes = [
    'akiec',
    'bcc',
    'bkl',
    'df',
    'mel',
    'nv',
    'vasc'
]


st.title("Skin Lesion Classifier")

st.write("Upload a skin lesion image to predict disease type.")


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    
    img = Image.open(uploaded_file)

    
    st.image(img, caption="Uploaded Image", use_column_width=True)

    
    img = img.resize((224, 224))

    
    img_array = image.img_to_array(img)

    
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Predicted class
    predicted_class = classes[np.argmax(prediction)]

    # Confidence
    confidence = np.max(prediction) * 100

    # Output
    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")