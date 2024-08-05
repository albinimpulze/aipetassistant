# Talking Animal Avatar App

This application allows users to generate a talking animal avatar based on a text prompt. It utilizes various libraries for image generation, speech recognition, and video creation.

## Features

- Generate images using a text prompt.
- Convert speech to text.
- Generate audio from text.
- Create a talking avatar video with lip-syncing.

## Requirements

To run this application, you need to install the following Python packages. You can do this by creating a `requirements.txt` file with the following content:

```plaintext
streamlit
numpy
Pillow
moviepy
phi
python-dotenv
SpeechRecognition
gTTS
glob2
requests
opencv-python
```

Running the Application
Ensure you have Python installed (preferably Python 3.7 or higher).
Clone this repository or download the code files.
Navigate to the project directory in your terminal.
Install the required packages using the command mentioned above.
Run the Streamlit application with the following command:
bash
Download
Copy code
streamlit run app.py
Replace app.py with the name of your main Python file if it's different.

Usage
Open your web browser and go to http://localhost:8501.
Enter a prompt for image generation in the input box.
Click the "Start Listening" button to speak your input.
The application will generate a response, convert it to speech, and create a talking avatar video.
