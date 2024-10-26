import os
import keras
from keras.models import load_model  # type: ignore
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Define username and password
USERNAME = "fer1cas"
PASSWORD = "0000"

# Streamlit login form
def login():
    st.sidebar.header("Login")
    
    # Check if the user is already logged in
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return True

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            return True
        else:
            st.sidebar.error("Invalid username or password")
            return False
    return False

# Main application
if login():
    st.header('Boiler Classification FER1CAS Model (Test Version)')
    flower_names = [
        'Boiler Oxygene Corrosion', 
        'Bride de raccordement Clogged due scale and TH value', 
        'Combusion chamber Cologged due bad combustion', 
        'Conductivity Sensor Clogged with scale',
        'Conductivity Sensor in normal condition', 
        'Deasalting Valve in normal condition',
        'Flame tube Clogged by scale due to TH ', 
        'Fuel Liquide in burner side',
        'Furnace corroded due to Dissolved Gases Oxygene and carbon',
        'Level Indicator damaged due to corrosion and hight PH', 
        'Level Indicator in normal condition', 
        'Level Probe Clogged',
        'Level Probe in normal condition', 
        'Low low Level electrode Clogged and corroded',
        'Low low Level electrode in normal condition', 
        'PT100 Clogged and corroded',
        'PT100 in a normal condition', 
        'Safety valve in normal condition', 
        'Valve Corroded'
    ]

    try:
        model = load_model('C:/Users/LENOVO/Desktop/my_model3boiler classification .keras')
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

    def classify_images(image):
        try:
            input_image = image.resize((480, 480))
            input_image_array = tf.keras.utils.img_to_array(input_image)
            input_image_exp_dim = tf.expand_dims(input_image_array, 0)

            predictions = model.predict(input_image_exp_dim)
            result = tf.nn.softmax(predictions[0])
            predicted_class = flower_names[np.argmax(result)]
            outcome = f'The Image belongs to   {predicted_class}   with a score of  {np.max(result) * 100:.2f}%'

            # Path to the Excel file
            file_path = 'C:/Users/LENOVO/Desktop/Boiler classfication/List comment boiler Pictures.xlsx'

            # Read the Excel file
            df = pd.read_excel(file_path)

            # Value to search in the "Categories" column
            valeur_recherchee = predicted_class  # Use the predicted class

            # Filter the rows that match the searched value
            resultat = df[df['Categories'] == valeur_recherchee]

            # Display results
            if not resultat.empty:
                st.write("Lignes correspondantes :")
                st.dataframe(resultat)

                # Fetch Cause and Recommendation from the filtered DataFrame
                cause_text = resultat['Causes'].values[0] if 'Causes' in resultat.columns else "Cause information not available."
                recommendation_text = resultat['Recommendation'].values[0] if 'Recommendation' in resultat.columns else "Recommendation information not available."
                
                # Display Cause and Recommendation
                #st.write("Cause:", cause_text)
                #st.write("Recommendation:", recommendation_text)
            else:
                st.write("For more informaion please contact an expert .")

            return outcome
        except Exception as e:
            st.error(f"Error during image classification: {e}")
            return "Error in classification"

    uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        try:
            # Read the image directly from the uploaded file
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Classify the image
            outcome = classify_images(image)

            # Display the classification outcome directly after the image
            st.markdown(outcome)

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
