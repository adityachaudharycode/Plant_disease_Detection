import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
st.title("Welcome to KrishiSahayak")
st.title("Crop Disease Detection")

st.write("Upload an image of your crop to detect diseases and get treatment suggestions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    try:
        model = tf.keras.models.load_model('C:/Users/adity/Desktop/run jupyter/data/model.h5')

        def preprocess_image(image):
            img = image.resize((225, 225))  
            img = np.array(img) / 255.0   
            img = np.expand_dims(img, axis=0)  
            return img

        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)

        st.write(f"Predictions: {predictions}")

        predicted_class = np.argmax(predictions)  
        confidence_score = np.max(predictions)  
# The data for cure is taken from data analyst i.e. Dr. MahaSweta bhowmick dutta
        disease_dict = {
            0: {"name": "early_blight_potato", "treatment": """
• Preventive measure\n
a) Use of resistant variety like: Kufri Garima, Kufri Gouralo, Kufri Badsa, Kufri Surya, Kufri Khayati, Kufri Himalini, etc.\n
b) Seed treatment with Mancozeb @ 2-2.5g/litre of water. Soak potato seed tuber in this solution for 10-15 minutes, keep it in a shade for drying, and after drying, planting should be done.\n
• Chemical control\n
a) Spray Mancozeb 2.5g/litre of H2O at the start of symptoms. If not controlled, spray Chlorothalonyl 2g/litre of water 2-3 times at 10 days interval or Carbenalazim + Mancozel mixture 2g/litre of H2O, 2-3 times at 10 days interval.\n
• Bio-control\n
a) Trichoderma viride can reduce disease intensity in the field.\n
b) Azospirillum Lipoterum strain AL-3 strain can help to control early blight.\n
c) T. harzianum and P. fluorescens can be used together to reduce disease intensity.\n
• Indian Method\n
a) Crop rotation helps to reduce the initial level of disease.\n
b) Use of disease-free seed tubers at planting time.\n
c) Clean cultivation during the entire season, especially at harvest.\n
d) Remove infected tubers before storing to prevent the spread of disease in storage.\n
"""
},
            1: {"name": "late_blight_potato", "treatment": """"
•	Preventive measure \n
a)	Use of resistant variety e.g. Kufri Jyoti, Kufri Himalini, Kufri Chipsona, Kufri Surya, Kufri Badsa, Kufri Giriraj, Kufri Sailaja , Kufri Khyati etc\n
b)	Late blight from regions must be planted with early varities of Potato\n
c)	Seed treatment with Mancozel @ 2-2.5 g/ litre of H2O. Soak potato seed tuber in this solution for 10-15 minutes, dry in a shade before planting \n
•	Chemical control\n
a)	At 40-45 days plant age, Mancozel 2-2.5 g/litre of water before starting of the symptom. Next, Cymoxanil (8%) + Mancozel (64%) mixture 3g/litre of water \n
b)	Or, Finamidon (1g) + Mancozel (2g) mixture 3g/litre of water\n
c)	Or, Dimethomorph (1g) + Mancozel (2g) mixture 3g/litre of water. Spraying must be done at the starting of symptom\n
d)	Lastly Mancozel 2-2.5 g/litre of water spraying continue 2-3 times at 7 days interval\n
•	Bio-control\n
a)	Bacillus, Pseudomonas, Streptomyces bacteria can be used as biological control agents for the control of the disease \n
b)	Trichoderma viride, Pennicillium viridicatum significantly reduce the disease infestation\n
•	Indian method\n
a)	Crop rotation with a non-solanaceous crop for at least 3 years\n
b)	Use of disease free seed tuber at planting time\n
c)	Control weeds throughout the seasons\n
d)	Manage irrigation to avoid prolonged period of wetness\n
e)	Keep more raw spacing or trimming dense leaves to promote more air circulation\n
f)	Remove infected tuber before storing of potato."""},
            2: {"name": "healthy_potato", "treatment": "No treatment needed, your crop is healthy!"}
        }

        disease_info = disease_dict.get(predicted_class, {"name": "Unknown", "treatment": "No treatment available"})

        st.write(f"*Predicted Disease:* {disease_info['name']}")
        st.write(f"*Confidence:* {confidence_score:.2f}")
        st.write(f"*Treatment:* {disease_info['treatment']}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
