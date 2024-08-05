import os
import streamlit as st
import numpy as np
from PIL import Image
import io
from moviepy.editor import ImageSequenceClip, AudioFileClip
from phi.assistant import Assistant
from phi.tools.duckduckgo import DuckDuckGo
from phi.llm.groq import Groq
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import tempfile
import glob
import requests
import cv2
import time  # Import the time module

# Load environment variables
load_dotenv()

# Create a static directory if it doesn't exist
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)

# Clean up old static files
def cleanup_static_files():
    for pattern in ["static/streamlit_audio_*.mp3", "static/streamlit_video_*.mp4"]:
        for filename in glob.glob(pattern):
            try:
                os.remove(filename)
            except Exception as e:
                st.warning(f"Failed to remove {filename}: {e}")

# Initialize the LLM and Assistant
llm = Groq(model="llama3-70b-8192", api_key=os.getenv("API_KEY"))
assistant = Assistant(llm=llm, tools=[DuckDuckGo()], show_tool_calls=False)

# Streamlit app title
st.title("Talking Animal Avatar App")
st.write("A front-facing image of a monkey posing for the camera, 3D talking tom style, standing")

# Define the API URL and headers for Hugging Face
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}

# Function to query the image generation API
@st.cache_data
def query_image_api(prompt):
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        st.error(f"Error querying image API: {e}")
        return None

# Image generation based on user prompt
image_prompt = st.text_input("Enter a prompt for image generation:")
generated_image = None

if image_prompt:
    with st.spinner("Generating image..."):
        image_bytes = query_image_api(image_prompt)
        if image_bytes:
            generated_image = Image.open(io.BytesIO(image_bytes))
            st.image(generated_image, caption="Generated Image", use_column_width=True)

# Function to listen to user input
@st.cache_resource
def get_recognizer():
    return sr.Recognizer()

def listen_to_user():
    recognizer = get_recognizer()
    try:
        with sr.Microphone() as source:
            st.write("Listening... Speak now!")
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand that.")
    except sr.RequestError:
        st.error("Sorry, there was an error with the speech recognition service.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    return None

# Function to generate response using the Assistant
def generate_response(prompt):
    try:
        response = assistant.run(f"Give up to 50 words only, conversation style. {prompt}")
        return ' '.join(response) if hasattr(response, '__iter__') else str(response)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

# Function to convert text to speech
def text_to_speech(text):
    try:
        if not isinstance(text, str):
            text = ' '.join(text) if hasattr(text, '__iter__') else str(text)
        
        audio_path = os.path.join(static_dir, f"streamlit_audio_{int(time.time())}.mp3")
        tts = gTTS(text=text, lang='en')
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None

# Function to create a simple mouth animation using OpenCV
def create_mouth_animation(image, audio_duration):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(cv_image, 1.1, 4)

    frames = []
    fps = 30
    total_frames = int(audio_duration * fps)

    for i in range(total_frames):
        frame = cv_image.copy()
        for (x, y, w, h) in faces:
            mouth_y = y + int(h * 0.7)
            mouth_h = int(h * 0.2)
            mouth_open = int(10 * np.sin(i * 0.5) + 10)
            cv2.ellipse(frame, (x + w//2, mouth_y + mouth_h//2), (w//4, mouth_open), 0, 0, 180, (0, 0, 0), -1)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    return frames

# Function to create talking avatar video with lip-sync
def create_talking_avatar(image, audio_path):
    try:
        video_path = os.path.join(static_dir, f"streamlit_video_{int(time.time())}.mp4")
        audio_clip = AudioFileClip(audio_path)
        frames = create_mouth_animation(image, audio_clip.duration)
        video_clip = ImageSequenceClip(frames, fps=30)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(video_path, codec='libx264', audio_codec='aac')
        return video_path
    except Exception as e:
        st.error(f"Error creating video: {str(e)}")
        return None

# Function to perform lip-sync
def lip_sync(audio_url, video_url):
    url = "https://api.synclabs.so/lipsync"
    payload = {
        "audioUrl": audio_url,
        "videoUrl": video_url,
        "maxCredits": 123,
        "model": "sync-1.6.0",
        "synergize": True,
        "pads": [0, 5, 0, 0],
        "synergizerStrength": 1,
        "webhookUrl": url
    }
    headers = {
        "x-api-key": "fe0da744-3a95-45d6-a11d-8e5ee43c7466",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        st.success("Lip-sync operation successful!")
        return response.json()  # Return the response for further processing
    else:
        st.error(f"Error: {response.status_code} {response.text}")
        return None

# Start listening button
if st.button("Start Listening"):
    user_input = listen_to_user()
    if user_input:
        response = generate_response(user_input)
        if response:
            st.write(f"Generated response: {response}")
            audio_file = text_to_speech(response)
            if audio_file:
                st.audio(audio_file)
                if generated_image is not None:
                    with st.spinner("Creating talking avatar video..."):
                        video_path = create_talking_avatar(generated_image, audio_file)
                        if video_path:
                            st.video(video_path)
                            # Perform lip-syncing
                            lip_sync_response = lip_sync(audio_file, video_path)
                            if lip_sync_response:
                                st.write("Lip-sync response:", lip_sync_response)

# Cleanup static files after the entire process
cleanup_static_files()