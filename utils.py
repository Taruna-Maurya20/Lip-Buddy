#                                          YHI CODE CHALEGA 
# import tensorflow as tf 
# from typing import List
# import cv2
# import os
# import numpy as np
# import imageio

# # Vocabulary definition
# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
# char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# # Mapping integers back to original characters
# num_to_char = tf.keras.layers.StringLookup(
#     vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
# )

# # Function to load video frames
# def load_video(path:str) -> List[np.ndarray]: 
#     cap = cv2.VideoCapture(path)
    
#     if not cap.isOpened():
#         print(f"Error: Cannot open video file {path}")
#         return None

#     frames = []
#     for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Error: Unable to read frame from {path}")
#             break
        
#         # Convert to grayscale and crop
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = frame[190:236,80:220]

#         # Normalize and convert to uint8 (0-255)
#         frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        
#         # Convert to 3-channel grayscale image (needed for GIF)
#         frame = np.stack([frame] * 3, axis=-1)
#         frames.append(frame)

#     cap.release()

#     if not frames:
#         print(f"Error: No frames read from {path}")
#         return None

#     return frames
    
# # Function to load alignments from .align file
# def load_alignments(path:str) -> List[str]: 
#     if not os.path.exists(path):
#         print(f"Error: Alignment file not found: {path}")
#         return None

#     with open(path, 'r') as f: 
#         lines = f.readlines() 

#     tokens = []
#     for line in lines:
#         line = line.split()
#         if len(line) >= 3 and line[2] != 'sil': 
#             tokens = [*tokens, ' ', line[2]]

#     if not tokens:
#         print(f"Error: No valid tokens found in alignment file {path}")
#         return None

#     return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# # Function to load video and corresponding alignments
# def load_data(path: str): 
#     path = bytes.decode(path.numpy())
#     file_name = path.split('/')[-1].split('.')[0]
#     file_name = path.split('\\')[-1].split('.')[0]

#     base_dir = 'C:/Users/TARUNA MAURYA/OneDrive/Desktop/Major/LipNet/data'
#     video_path = os.path.join(base_dir, 's1', f'{file_name}.mpg')
#     alignment_path = os.path.join(base_dir, 'alignments', 's1', f'{file_name}.align')

#     print(f"Video path: {video_path}")
#     print(f"Alignment path: {alignment_path}")

#     if not os.path.exists(video_path):
#         print(f"Error: Video file not found: {video_path}")
#         return None, None

#     frames = load_video(video_path)
#     if frames is None:
#         print(f"Error: Failed to load video data from {video_path}")
#         return None, None

#     if not os.path.exists(alignment_path):
#         print(f"Error: Alignment file not found: {alignment_path}")
#         return frames, None

#     alignments = load_alignments(alignment_path)
#     if alignments is None:
#         print(f"Error: Failed to load alignment data from {alignment_path}")
#         return frames, None

#     return frames, alignments

# # Main script example
# if __name__ == "__main__":

#     file_path = tf.convert_to_tensor("C:/Users/TARUNA MAURYA/OneDrive/Desktop/Major/LipNet/data/s1")
#     video, annotations = load_data(file_path)

#     if video is None:
#         print("Video loading failed.")
#     if annotations is None:
#         print("Alignment loading failed or missing.")
    
#     # Save video as GIF
#     if video:
#         imageio.mimsave('animation.gif', video, fps=10)
#         print("GIF saved as 'animation.gif'")






















# import tensorflow as tf
# from typing import List
# import cv2
# import os 

# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
# char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# # Mapping integers back to original characters
# num_to_char = tf.keras.layers.StringLookup(
#     vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
# )

# def load_video(path:str) -> List[float]: 
#     cap = cv2.VideoCapture(path)
#     frames = []
#     for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
#         ret, frame = cap.read()
#         frame = tf.image.rgb_to_grayscale(frame)
#         frames.append(frame[190:236,80:220,:])
#     cap.release()
    
#     mean = tf.math.reduce_mean(frames)
#     std = tf.math.reduce_std(tf.cast(frames, tf.float32))
#     return tf.cast((frames - mean), tf.float32) / std
    
# def load_alignments(path: str) -> List[str]:
#     with open(path, 'r') as f:
#         lines = f.readlines()
#     tokens = []
#     for line in lines:
#         line = line.split()
#         if line[2] != 'sil':
#             tokens = [*tokens, ' ', line[2]]
    
#     # Convert tokens to a RaggedTensor
#     ragged_tokens = tf.strings.unicode_split(tokens, input_encoding='UTF-8')
    
#     # Convert the RaggedTensor to a dense tensor
#     dense_tokens = ragged_tokens.to_tensor(default_value='')
    
#     # Apply char_to_num and slice the tensor
#     return char_to_num(dense_tokens)[1:]

# def load_data(path: str): 
#     path = bytes.decode(path.numpy())
#     # File path splitting for Windows
#     file_name = path.split('\\')[-1].split('.')[0]
#     video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
#     alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
#     frames = load_video(video_path) 
#     alignments = load_alignments(alignment_path)
    
#     return frames, alignments





























import tensorflow as tf
from typing import List
import cv2
import os
import numpy as np
import imageio

# Vocabulary definition
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Function to load video frames
def load_video(path:str) -> List[np.ndarray]: 
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {path}")
        return None

    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read frame from {path}")
            break
        
        # Convert to grayscale and crop
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[190:236,80:220]

        # Normalize and convert to uint8 (0-255)
        frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        
        # Convert to 3-channel grayscale image (needed for GIF)
        frame = np.stack([frame] * 3, axis=-1)
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"Error: No frames read from {path}")
        return None

    return frames
    
# Function to load alignments from .align file
def load_alignments(path:str) -> List[str]: 
    if not os.path.exists(path):
        print(f"Error: Alignment file not found: {path}")
        return None

    with open(path, 'r') as f: 
        lines = f.readlines() 

    tokens = []
    for line in lines:
        line = line.split()
        if len(line) >= 3 and line[2] != 'sil': 
            tokens = [*tokens, ' ', line[2]]

    if not tokens:
        print(f"Error: No valid tokens found in alignment file {path}")
        return None

    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# Function to load video and corresponding alignments
def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    file_name = path.split('\\')[-1].split('.')[0]

    base_dir = 'C:\\Users\\TARUNA MAURYA\\OneDrive\\Desktop\\Major\\LipNet\\data'
    video_path = os.path.join(base_dir, 's1', f'{file_name}.mpg')
    alignment_path = os.path.join(base_dir, 'alignments', 's1', f'{file_name}.align')

    print(f"Video path: {video_path}")
    print(f"Alignment path: {alignment_path}")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None, None

    frames = load_video(video_path)
    if frames is None:
        print(f"Error: Failed to load video data from {video_path}")
        return None, None

    if not os.path.exists(alignment_path):
        print(f"Error: Alignment file not found: {alignment_path}")
        return frames, None

    alignments = load_alignments(alignment_path)
    if alignments is None:
        print(f"Error: Failed to load alignment data from {alignment_path}")
        return frames, None

    return frames, alignments

# Main script example
if __name__ == "__main__":
    file_path = tf.convert_to_tensor("C:\\Users\\TARUNA MAURYA\\OneDrive\\Desktop\\Major\\LipNet\\data\\s1\\sample.mpg")
    video, annotations = load_data(file_path)

    if video is None:
        print("Video loading failed.")
    if annotations is None:
        print("Alignment loading failed or missing.")
    
    # Save video as GIF
    if video:
        imageio.mimsave('animation.gif', video, fps=10)
        print("GIF saved as 'animation.gif'")
