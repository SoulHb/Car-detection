import streamlit as st
import requests
import os
import numpy as np
from PIL import Image
from io import BytesIO

# Determine the API URL based on the execution environment
if "DOCKER_MODE" in os.environ:
    api_url = "http://api:5000"
else:
    api_url = "http://localhost:5000"

# Add some content to the app
st.title('Car Detection')
st.write('Upload an image of a cars and click the "Detect" button')
input_type = st.selectbox("Select the input type", ("Image", "Video"))
if input_type == "Image":
    # Upload the input image file
    uploaded_file = st.file_uploader("Choose file...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        bytes_image = BytesIO()
        image.save(bytes_image, format="JPEG")
        bytes_image = bytes_image.getvalue()
        s = st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button('Detect'):
            s.empty()
            predicted = requests.post("http://127.0.0.1:5000/image_predict", files={'file': bytes_image})
            predicted_data = predicted.json()
            predicted = np.array(predicted_data['prediction'])
            st.image(predicted, caption='Predicted Image', use_column_width=True)

elif input_type == "Video":
    input_video = st.file_uploader("Upload a video file", type=["mp4", "rb", "webm", 'avi'])
    if input_video is not None:
        video = input_video.getvalue()
        v = st.video(video)
        if st.button('Run'):
            v.empty()
            t = st.empty()
            t.markdown('Running...')
            predicted = requests.post(f"{api_url}/process_video", files={'file': input_video})
            predicted = predicted.content
            st.video(predicted)