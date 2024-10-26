import os
import keras
from keras.models import load_model # type: ignore
import streamlit as st 
import tensorflow as tf
import numpy as np



st.header('Boiler Classification CNN Model')
flower_names = ['Boiler Oxygene Corrosion', 'Bride de raccordement Clogged due scale and TH value', 'Combusion chamber Cologged due bad combustion', 'Conductivity Sensor Clogged with scale','Conductivity Sensor in normal condition','Deasalting Valve in normal condition','Flame tube Clogged by scale due to TH ','Fuel Liquide in burner side','Furnace corroded due to Dissolved Gases Oxygene and carbon','Level Indicator damaged due to corrosion and hight PH','Level Indicator in normal condition','Level Probe Clogged','Level Probe in normal condition','Low low Level electrode Clogged and corroded','Low low Level electrode in normal condition','PT100 Clogged and corroded','PT100 in a normal condition','Safety valve in normal condition','Valve Corroded']

model = load_model('C:/Users/LENOVO/Desktop/Boiler classfication/my_modelboiler.keras')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('C:/Users/LENOVO/Desktop/Boiler classfication//upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width = 200)

    st.markdown(classify_images(uploaded_file))

