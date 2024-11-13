import pandas as pd
import cv2
import numpy as np
from transformers import pipeline
from deepface import DeepFace
import plotly.graph_objects as go
from collections import deque
import tempfile
import torch
import subprocess
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
from transformers import AutoTokenizer
import streamlit as st
import time

# Path to FFmpeg executable
FFMPEG_PATH = "C:\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe"
os.environ["FFMPEG_BINARY"] = FFMPEG_PATH

# Create a temporary directory for audio chunks
TEMP_DIR = tempfile.mkdtemp()

# Initialize tokenizer and models
emotion_tokenizer = AutoTokenizer.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion')
emotion_classifier = pipeline(
    'text-classification',
    model='bhadresh-savani/distilbert-base-uncased-emotion',
    return_all_scores=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
speech_recognizer = pipeline(
    'automatic-speech-recognition',
    model='openai/whisper-medium',
    device=device
)

class ClassroomMonitor:
    def __init__(self):
        self.engagement_history = deque([0.7] * 10, maxlen=100)
        self.curriculum_adherence = deque([0.5] * 10, maxlen=100)
        self.emotion_stats = {
            'joy': 1, 'sadness': 1, 'anger': 1, 
            'fear': 1, 'love': 1, 'surprise': 1
        }
        self.transcribed_text = []
        self.frame_count = 0
        self.last_update_time = time.time()
        self.update_interval = 1.0

    def split_audio(self, audio_path, chunk_length_ms=30000):
        """Split audio into chunks with proper temporary file handling."""
        try:
            # Load the audio file
            audio = AudioSegment.from_file(audio_path, format="wav")
            chunks = make_chunks(audio, chunk_length_ms)
            chunk_paths = []
            
            # Create chunks in the temporary directory
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(TEMP_DIR, f"chunk_{i}.wav")
                chunk.export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)
                
            return chunk_paths
        except Exception as e:
            st.error(f"Error splitting audio: {str(e)}")
            return []

    def analyze_speech(self, audio_path, curriculum_topics):
        """Analyze speech with improved file handling."""
        try:
            chunk_paths = self.split_audio(audio_path)
            if not chunk_paths:
                return 0.5, 'neutral', "Error processing audio"

            full_transcription = []
            
            for chunk_path in chunk_paths:
                try:
                    # Verify file exists before processing
                    if os.path.exists(chunk_path):
                        transcription = speech_recognizer(chunk_path, return_timestamps=True)
                        full_transcription.append(transcription['text'])
                except Exception as e:
                    st.warning(f"Error processing chunk {chunk_path}: {str(e)}")
                finally:
                    # Clean up chunk file if it exists
                    if os.path.exists(chunk_path):
                        try:
                            os.remove(chunk_path)
                        except Exception:
                            pass

            text = " ".join(full_transcription)
            if not text.strip():
                return 0.5, 'neutral', "No text transcribed"

            # Calculate curriculum adherence
            text_lower = text.lower()
            topic_mentions = sum(
                sum(word.lower() in text_lower 
                    for word in topic.split())
                for topic in curriculum_topics
            )
            total_possible_mentions = sum(len(topic.split()) for topic in curriculum_topics)
            adherence_score = min(topic_mentions / max(total_possible_mentions, 1), 1.0)
            
            # Analyze emotions
            emotion_results = emotion_classifier([text])
            dominant_emotion = max(emotion_results[0], key=lambda x: x['score'])
            
            # Store transcription
            self.transcribed_text.append(text)
            if len(self.transcribed_text) > 5:
                self.transcribed_text.pop(0)
                
            return adherence_score, dominant_emotion['label'], text
            
        except Exception as e:
            st.error(f"Speech analysis error: {str(e)}")
            return 0.5, 'neutral', "Error analyzing speech"

    def analyze_frame(self, frame):
        """Analyze frame emotions with error handling."""
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if analysis and isinstance(analysis, list) and len(analysis) > 0:
                emotions = analysis[0]['emotion']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                return dominant_emotion.lower()
        except Exception:
            pass
        return None

    def update_stats(self, adherence, emotion):
        """Update monitoring statistics."""
        current_time = time.time()
        
        if current_time - self.last_update_time >= self.update_interval:
            base_engagement = adherence * 0.7 + np.random.normal(0.5, 0.1) * 0.3
            engagement = max(0.0, min(1.0, base_engagement))
            self.engagement_history.append(engagement)
            self.curriculum_adherence.append(adherence)
            
            if emotion:
                self.emotion_stats[emotion.lower()] = self.emotion_stats.get(emotion.lower(), 0) + 1
            
            self.last_update_time = current_time
            self.frame_count += 1

def extract_audio_from_video(video_path):
    """Extract audio with proper temporary file handling."""
    try:
        video = VideoFileClip(video_path)
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP_DIR)
        audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        video.close()
        return audio_path
    except Exception as e:
        st.error(f"Audio extraction error: {str(e)}")
        return None

def cleanup_temp_files():
    """Clean up temporary files."""
    try:
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.warning(f"Error deleting {file_path}: {str(e)}")
    except Exception as e:
        st.error(f"Error cleaning up temporary files: {str(e)}")

def main():
    st.set_page_config(layout="wide")
    st.title("🎓 Real-time Classroom Monitor with Speech Analysis")
    
    if 'monitor' not in st.session_state:
        st.session_state.monitor = ClassroomMonitor()
        st.session_state.monitoring = False
        st.session_state.curriculum = [
            "Introduction to Python",
            "Variables and Data Types",
            "Control Structures",
            "Functions"
        ]
    
    with st.sidebar:
        st.header("📊 Monitoring Settings")
        curriculum = st.text_area(
            "Enter curriculum topics (one per line)",
            "\n".join(st.session_state.curriculum)
        )
        st.session_state.curriculum = [x.strip() for x in curriculum.split("\n") if x.strip()]
        
        if st.button("Toggle Monitoring", key="toggle"):
            st.session_state.monitoring = not st.session_state.monitoring
            
        monitoring_status = "🟢 Active" if st.session_state.monitoring else "🔴 Inactive"
        st.write(f"Monitoring Status: {monitoring_status}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Upload Classroom Video")
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_file:
            try:
                # Create temporary video file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=TEMP_DIR) as temp_video_file:
                    video_path = temp_video_file.name
                    temp_video_file.write(uploaded_file.read())
                
                # Process audio
                audio_path = extract_audio_from_video(video_path)
                if audio_path:
                    adherence, emotion, text = st.session_state.monitor.analyze_speech(
                        audio_path, 
                        st.session_state.curriculum
                    )
                    if text:
                        st.write("Transcribed Text:", text)
                    
                    # Video processing
                    frame_placeholder = st.empty()
                    cap = cv2.VideoCapture(video_path)
                    
                    try:
                        while cap.isOpened() and st.session_state.monitoring:
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                            
                            visual_emotion = st.session_state.monitor.analyze_frame(frame)
                            st.session_state.monitor.update_stats(adherence, visual_emotion)
                            
                            time.sleep(0.033)
                            
                    finally:
                        cap.release()
                        
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                
            finally:
                # Clean up all temporary files
                cleanup_temp_files()
    
    with col2:
        st.header("📈 Real-time Metrics")
        
        engagement_value = float(np.mean(list(st.session_state.monitor.engagement_history)))
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=engagement_value * 100,
            title={'text': "Class Engagement"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        adherence = float(np.mean(list(st.session_state.monitor.curriculum_adherence)))
        st.metric("Curriculum Adherence", f"{adherence:.1%}")
        
        st.subheader("Emotional Climate")
        emotion_df = pd.DataFrame(
            list(st.session_state.monitor.emotion_stats.items()),
            columns=['Emotion', 'Count']
        )
        st.bar_chart(emotion_df.set_index('Emotion'))
        
        if st.session_state.monitor.transcribed_text:
            st.subheader("Recent Speech Transcripts")
            for text in st.session_state.monitor.transcribed_text[-3:]:
                st.text(text)

        st.header("🔔 Real-time Alerts")
        if adherence < 0.6:
            st.warning("⚠️ Curriculum coverage is below target (60%)")
        if engagement_value < 0.7:
            st.warning("⚠️ Student engagement is low (below 70%)")

if __name__ == "__main__":
    main()