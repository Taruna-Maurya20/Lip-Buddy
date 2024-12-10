# # Import all of the dependencies
# import streamlit as st
# import os 
# import imageio 

# import tensorflow as tf 
# from utils import load_data, num_to_char
# from modelutil import load_model

# #chatgpt
# import os
# print("Current working directory:", os.getcwd())

# # Set the layout to the streamlit app as wide 
# st.set_page_config(layout='wide')

# # Setup the sidebar
# with st.sidebar: 
#     st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
#     st.title('LipBuddy')
#     st.info('This application is originally developed from the LipNet deep learning model.')

# st.title('LipNet Full Stack App') 


# import os
# folder_path = os.path.join('..', 'data', 's1')
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)
#     print(f"Created missing folder: {folder_path}")
# else:
#     options = os.listdir(folder_path)
#     print("Folder contents:", options)

# import os

# folder_path = os.path.join('..', 'data', 's1')
# if os.path.exists(folder_path):
#     options = os.listdir(folder_path)
#     print("Folder contents:", options)
# else:
#     print(f"Folder {folder_path} does not exist.")

# # Generating a list of options or videos 
# import os #chatgpt
# folder_path = 'C://Users//TARUNA MAURYA//OneDrive//Desktop//Major//LipNet//data//s1'
# options = os.listdir(folder_path) #endof chatgpt
# #options = os.listdir(os.path.join('..', 'data', 's1'))
# selected_video = st.selectbox('Choose video', options)



# # Generate two columns 
# col1, col2 = st.columns(2)

# if options: 

#     # Rendering the video 
#     with col1: 
#         st.info('The video below displays the converted video in mp4 format')
#         file_path = os.path.join('..','data','s1', selected_video)
#         os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

#         # Rendering inside of the app
#         video = open('test_video.mp4', 'rb') 
#         video_bytes = video.read() 
#         st.video(video_bytes)


#     with col2: 
#         st.info('This is all the machine learning model sees when making a prediction')
#         video, annotations = load_data(tf.convert_to_tensor(file_path))
#         imageio.mimsave('animation.gif', video, fps=10)
#         st.image('animation.gif', width=400) 

#         st.info('This is the output of the machine learning model as tokens')
#         model = load_model()
#         yhat = model.predict(tf.expand_dims(video, axis=0))
#         decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
#         st.text(decoder)

#         # Convert prediction to text
#         st.info('Decode the raw tokens into words')
#         converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
#         st.text(converted_prediction)
        























                        #YAHI WALA CODE CHALEGA
# import streamlit as st
# import os
# import imageio
# import tensorflow as tf
# from utils import load_data, num_to_char
# from modelutil import load_model

# # Set layout
# st.set_page_config(layout='wide')

# with st.sidebar:
#     st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
#     st.title("About")
#     st.info("Focuses on decoding text from a speaker's mouth movements. By using deep learning techniques and advanced models, the project achieves impressive accuracy in mapping video frames to text.")

# st.title('LipNet 🗣️')
# # Retrieve the list of videos
# videos = os.listdir(os.path.join('..', 'data', 's1'))
# selected_video = st.selectbox('Pick a video', videos)

# # Generate 2 columns
# col1, col2 = st.columns(2)

# if videos:
#     with col1:
#         st.info('Chosen video:')
#         file_path = os.path.join('..', 'data', 's1', selected_video)
#         os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

#         # Rendering
#         video = open('test_video.mp4', 'rb')
#         video_bytes = video.read()
#         st.video(video_bytes)
    
#     with col2:
#         hehe = st.info('Input for model:')
#         video, annotations = load_data(tf.convert_to_tensor(file_path))
#         imageio.mimsave('animation.gif', video, fps=10)
#         st.image('animation.gif', width=400)

#         st.info('Tokenized prediction:')
#         model = load_model()
#         yhat = model.predict(tf.expand_dims(video, axis=0))
#         # Greedy algorithm takes most probable prediction
#         decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
#         st.text(decoder)

#         # Decode prediction
#         st.info('Decoded prediction into text:')
#         converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('UTF-8')
#         st.text(converted_prediction)
        













import streamlit as st
import os
import imageio
import cv2
import numpy as np
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')
# Generating a list of options or videos
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        
        # Convert each frame of the video to grayscale and ensure dtype is uint8
        gray_video = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video], dtype=np.uint8)

        # Save the grayscale video as a GIF for display
        imageio.mimsave('animation.gif', gray_video, fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        
        # Expand dimensions to add batch size and channel before prediction
        gray_video_expanded = gray_video[..., np.newaxis]  # Shape (75, 46, 140, 1)
        yhat = model.predict(tf.expand_dims(gray_video_expanded, axis=0))
        

        # Decode the model's prediction
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

