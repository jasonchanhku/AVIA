# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:56:52 2021

@author: JC17642
"""
import os
import streamlit as st
import asyncio
import logging
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import math
from datetime import datetime
import textwrap
import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
from aiortc.contrib.media import MediaPlayer

from gaze_tracking import GazeTracking

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

import tensorflow as tf
from tensorflow.keras.models import load_model

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# Here we are importing the already trained HaarCascade models

questions = [
    
    "Please introduce and give us a brief background of yourself",
    
    "What really motivates you to work?",
    
    "Tell us a situation where you really had to and compromise in your workplace",
    
    "Was there a time where you had to step up in leadership?",
    
    "What do you like and dislike about your current job?"
    
]

def draw_label(img, text, shape, bg_color):
    font = cv2.FONT_ITALIC
    font_size = 0.8
    color = (0, 0, 0)
    font_thickness = 1
    margin = 5
    
    wrapped_text = textwrap.wrap(text, width=40)
    
    for i, line in enumerate(wrapped_text):
        
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        
        gap = textsize[1] + 10
        
        x = int((img.shape[1] - textsize[0]) / 2) 
        y = int((img.shape[0] + textsize[1]) / 2) + i * gap - 190
        
        end_x = x + textsize[0] + margin
        end_y = y - textsize[1] - margin 
        
        cv2.rectangle(img, (x, y), (end_x, end_y), bg_color, cv2.FILLED)

        cv2.putText(img, line, (x, y), font,
                    font_size, 
                    color, 
                    font_thickness, 
                    cv2.LINE_AA)
    
    return textsize

def draw_warning(img, text, shape, bg_color):
    font = cv2.FONT_HERSHEY_COMPLEX
    font_size = 1
    color = (0, 0, 255)
    font_thickness = 2
    margin = 5
    
    wrapped_text = textwrap.wrap(text, width=30)
    
    for i, line in enumerate(wrapped_text):
        
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        
        gap = textsize[1] + 10
        
        x = int((img.shape[1] - textsize[0]) / 2) 
        y = int((img.shape[0] + textsize[1]) / 2) + i * gap
        
        end_x = x + textsize[0] + margin
        end_y = y - textsize[1] - margin 
        
        cv2.putText(img, line, (x, y), font,
                    font_size, 
                    color, 
                    font_thickness, 
                    cv2.LINE_AA)
    
    return textsize



# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def find_faces(img: np.ndarray, bgr=True) -> list:
    
    gray_image_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray_image_array,
        scaleFactor=1.3,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(100, 100),
    )

    return faces


def apply_offsets(face_coordinates):
    x, y, width, height = face_coordinates
    x_off, y_off = (10, 10)
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def tosquare(bbox):
    """Convert bounding box to square by elongating shorter side."""
    x, y, w, h = bbox
    if h > w:
        diff = h - w
        x -= diff // 2
        w += diff
    elif w > h:
        diff = w - h
        y -= diff // 2
        h += diff

    return (x, y, w, h)

def pad(image):
    row, col = image.shape[:2]
    bottom = image[row - 2 : row, 0:col]
    mean = cv2.mean(bottom)[0]

    bordersize = 40
    padded_image = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean],
    )
    return padded_image


def preprocess_input(x, v2=False):
    x = x.astype("float32")
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def detect_emotions(img):
    
    face_rectangles = find_faces(img, bgr=True)
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    emotions = []
    for face_coordinates in face_rectangles:
        face_coordinates = tosquare(face_coordinates)
        x1, x2, y1, y2 = apply_offsets(face_coordinates)
    
        if y1 < 0 or x1 < 0:
            gray_img = pad(gray_img)
            x1 += 40
            x2 += 40
            y1 += 40
            y2 += 40
            x1 = np.clip(x1, a_min=0, a_max=None)
            y1 = np.clip(y1, a_min=0, a_max=None)
    
        gray_face = gray_img[y1:y2, x1:x2]
    
        try:
            gray_face = cv2.resize(gray_face, (64, 64))
        except Exception as e:
            print("{} resize failed: {}".format(gray_face.shape, e))
            continue
        
        # Local Keras model
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
    
        emotion_prediction = best_model.predict(gray_face)[0]
        labelled_emotions = {
            emotion_labels[idx]: round(float(score), 2)
            for idx, score in enumerate(emotion_prediction)
        }
    
        emotions.append(
            dict(box=face_coordinates, emotions=labelled_emotions)
        )
                
    return emotions
        
        

def app(sesh):
    
    st.title('Stage 1 - Video Interview')
    st.write('''The video interview tests you on your facial expressions, eye contact, and posture while answering the questions. This is essential as you might not be conscious of your body language during an online interview. Try to be yourself during this process as it will make the performance review more genuine.
             Please kindly follow the instructions on the instructions panel on the sidebar.''')
    
    st.sidebar.markdown("## Step 1")
    
    st.sidebar.markdown('''Click "Select Device" to select your active webcam camera. After that, click "Start."''')

    st.sidebar.markdown("## Step 2")
    
    st.sidebar.markdown('''Adjust the angle of your frontal face such that your face is always detected. You may adjust the threshold below if necessary.''')
    
    st.sidebar.markdown("## Step 3")
    
    st.sidebar.markdown('''Begin answering the first question and then scroll the slider to the next question.''')

    st.sidebar.markdown("## Step 4")
    
    st.sidebar.markdown('''Click stop when finished recording and you may scroll back up to the top to move on the next session.''')
    
    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)
        
    app_sendonly_video()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")

face_detector = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")

emotion_labels = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "sad",
            5: "surprise",
            6: "neutral",
        }

best_model = load_model('./models/emotion_model.hdf5')
best_model.make_predict_function()
gaze = GazeTracking()
main_list = []

def app_sendonly_video():
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 1

    class Detection(NamedTuple):
        face_id: str
        datetime: str
        question: int
        name: str
        prob: float
        eye_contact: str

    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        confidence_threshold: int
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            #self.detector = FER()
            #self.best_model = load_model('emotion_detection_model_gsn.h5')
            #self.best_model = load_model('emotion_model.hdf5')
            #self.best_model.make_predict_function()
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, all_detections):
            # loop over the detections
            result: List[Detection] = []
            if len(all_detections) == 0:
                _ = draw_warning(image, "No face detected, please re-adjust position.", image.shape, (255,255,255))
                result.append(Detection(face_id=np.nan, datetime=np.nan,question=self.confidence_threshold, name=np.nan, prob=np.nan, eye_contact=np.nan))
            else:
                try:
                    for face_id, detections in enumerate(all_detections):
                        faces = np.array(detections['box']).reshape(1,4)
                        emotions = detections['emotions']
                        # Draw a rectangle around the faces
                        for (x, y, w, h) in faces:
                            
                            # rectangle around face
                            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # emotion prediction
                            emotion_pred = max(emotions, key=emotions.get)
                            emotion_prob = emotions[max(emotions, key=emotions.get)]
                            emotion_text = emotion_pred + ' ' + str(round(emotion_prob, 2))
                            cv2.putText(image, "emotion: " + emotion_text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            gaze.refresh(image)
                            
                            image = gaze.annotated_frame()
                            # eye gaze detection
                            text = ""

                            if gaze.is_blinking():
                                text = "blink"
                            elif gaze.is_right():
                                text = "right"
                            elif gaze.is_left():
                                text = "left"
                            elif gaze.is_center():
                                text = "center"
                        
                            cv2.putText(image, "eye contact: "+ text, (x , y - 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                        
                            left_pupil = gaze.pupil_left_coords()
                            right_pupil = gaze.pupil_right_coords()
                            #cv2.putText(image, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
                            #cv2.putText(image, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
                                                    
                            
                            result.append(Detection(face_id = face_id, datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),question=self.confidence_threshold, name=emotion_pred, prob=emotion_prob, eye_contact=text))
                            main_list.append(Detection(face_id = face_id, datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"),question=self.confidence_threshold, name=emotion_pred, prob=emotion_prob, eye_contact=text))
        
        
                except:
                    _ = draw_warning(image, "No face detected, please re-adjust position.", image.shape, (255,255,255))
                    result.append(Detection(face_id=np.nan, datetime=np.nan,question=self.confidence_threshold, name=np.nan, prob=np.nan, eye_contact=np.nan))

            return image, result
        
        

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            all_detections = detect_emotions(image)
            
            draw_label(image, questions[self.confidence_threshold - 1], image.shape, (255,255,255))

            annotated_image, result = self._annotate_image(image, all_detections)
            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=MobileNetSSDVideoProcessor,
        async_processing=True,
    )

    confidence_threshold = st.slider(
        "Question Number", 1, 5, DEFAULT_CONFIDENCE_THRESHOLD, 1
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

    st.subheader("Facial Expression & Eye Contact Prediction Result")
    
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            if webrtc_ctx.video_processor:
                try:
                    result = webrtc_ctx.video_processor.result_queue.get(
                        timeout=1.0
                    )
                except queue.Empty:
                    result = None
                labels_placeholder.table(result)
            else:
                break
            
    emotion_df = pd.DataFrame(main_list)
    emotion_df.to_csv('./output/emotion_data.csv', index=False)
