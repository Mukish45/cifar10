import streamlit as st
import cv2
import cv2
import numpy as np
from tensorflow import keras
model = keras.models.load_model('Mymodel.h5')

upload_image = st.file_uploader(label='Upload image', type=['png', 'jpg','jpeg'], accept_multiple_files=False)
img = cv2.resize(upload_image, dsize=(32, 32))
img = np.reshape(img,[1,32,32,3])
y_predict = np.argmax(model.predict(img), axis=1)
if upload_image is not None:
    st.text(y_predict)


