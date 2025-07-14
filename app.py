# app.py

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import hf_hub_download

# For real-time video processing
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av # Python bindings for FFmpeg (used by streamlit-webrtc)

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Live Face Mask Detection", layout="centered")
st.title("😷 Real-time Face Mask Detection")
st.markdown("---")

# --- 1. Load YOLOv11 Face Detection Model ---
@st.cache_resource
def load_yolo_face_model():
    with st.spinner("Loading YOLOv11 Face Detection Model..."):
        model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    return YOLO(model_path)

yolo_face_model = load_yolo_face_model()

# --- 2. Load Your Fine-tuned MobileNetV2 Mask Classification Model ---
@st.cache_resource
def load_mask_model():
    with st.spinner("Loading Mask Classification Model..."):
        return load_model("mask_detector_model.h5")

mask_model = load_mask_model()

# --- 3. Preprocessing Function for MobileNetV2 (MUST MATCH TRAINING!) ---
def preprocess_for_mobilenet(face_image):
    # CRITICAL FIX: Convert BGR to RGB if input is from OpenCV (which it is, after face cropping)
    # MobileNetV2 (pretrained on ImageNet) expects RGB color order.
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB) # ADDED/FIXED LINE
    
    face_resized = cv2.resize(face_rgb, (224, 224)) # Ensure resizing happens on RGB image
    face_array = img_to_array(face_resized)
    face_array = np.expand_dims(face_array, axis=0) # Add batch dimension for model input
    face_array = face_array / 255.0 # Normalize pixels to [0, 1]
    return face_array

# --- 4. Video Processing Class for Streamlit-WebRTC ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # Models are loaded via st.cache_resource, so they are only loaded once per session
        self.yolo_model = yolo_face_model
        self.mask_model = mask_model

        # Ensure this order matches the output from train_mask_classifier.py's class_indices:
        # Index 0 is 'with_mask', Index 1 is 'without_mask'.
        self.class_labels = ["With Mask", "No Mask"] # Correct mapping: 0 -> "With Mask", 1 -> "No Mask"

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        # Convert the incoming video frame from WebRTC to an OpenCV BGR NumPy array
        img_bgr = frame.to_ndarray(format="bgr24")

        # Perform YOLOv11 face detection on the current frame
        # conf=0.4 is a good balance; adjust if too many/few detections in live feed
        results = self.yolo_model.predict(source=img_bgr, conf=0.4, verbose=False)
        
        # Process each detected face
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                # Ensure bounding box coordinates are within the image dimensions
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_bgr.shape[1], x2)
                y2 = min(img_bgr.shape[0], y2)

                # Crop the face region using the detected bounding box
                face = img_bgr[y1:y2, x1:x2]

                # Skip if the cropped face region is invalid (e.g., zero width or height)
                if face.shape[0] == 0 or face.shape[1] == 0:
                    continue

                # Preprocess the cropped face for the MobileNetV2 classifier
                # This now includes the BGR to RGB conversion
                face_array_processed = preprocess_for_mobilenet(face) 

                # Predict mask presence using the MobileNetV2 model
                pred = self.mask_model.predict(face_array_processed, verbose=0)[0] # verbose=0 suppresses console output
                label_idx = np.argmax(pred) # Get the index of the highest probability
                confidence = pred[label_idx] # Get the confidence for that prediction

                # Get the human-readable label and assign color based on prediction
                label = self.class_labels[label_idx]
                # Green for "With Mask", Red for "No Mask"
                color = (0, 255, 0) if label.lower() == "with mask" else (0, 0, 255)

                # Draw the bounding box on the original frame
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

                # Prepare the text to display (e.g., "With Mask (99.5%)")
                display_text = f"{label} ({confidence*100:.1f}%)"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                # Calculate text size for background rectangle and positioning
                (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)
                text_x = x1
                text_y = y1 - 10 # Default: place text just above the bounding box
                
                # Ensure padding and baseline are considered to prevent cutoff
                if text_y - text_height - baseline < 0:
                     text_y = y1 + text_height + 5 + baseline # Place below the box if no space above
                
                # Draw a filled rectangle behind the text for better readability
                cv2.rectangle(img_bgr, (text_x, text_y - text_height - baseline),
                              (text_x + text_width, text_y), color, -1) # -1 means filled rectangle
                
                # Draw the text itself (white color for contrast)
                cv2.putText(img_bgr, display_text, (text_x, text_y - baseline),
                            font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # Return the processed frame to be displayed by Streamlit-WebRTC
        return img_bgr

# --- 5. Streamlit UI for Live Webcam Feed ---
st.header("Live Webcam Feed")
st.info("Click 'START' to activate your webcam and begin real-time mask detection.")

webrtc_streamer(
    key="mask-detection-webcam",
    mode=WebRtcMode.SENDRECV, # Allows browser to send video and receive processed video
    # RTC configuration for STUN servers helps with NAT traversal (important for deployment)
    rtc_configuration={ 
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_processor_factory=VideoProcessor, # Use our custom processor for frames
    media_stream_constraints={
        "video": True, # Request video stream from camera
        "audio": False # Do not request audio
    },
    async_processing=True # Process frames asynchronously for smoother performance
)

st.markdown("---")

# --- 6. Static Image Upload Section (Optional) ---
st.header("Static Image Upload (Optional)")
uploaded_file = st.file_uploader("Or, upload an image for detection", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open the uploaded image and convert to OpenCV format
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Run YOLOv11 face detection on the static image
    results = yolo_face_model.predict(source=img_np, conf=0.4, verbose=False)
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        class_labels_static = ["With Mask", "No Mask"] # Correct mapping: 0 -> "With Mask", 1 -> "No Mask"

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, x1) ; y1 = max(0, y1)
            x2 = min(img_bgr.shape[1], x2) ; y2 = min(img_bgr.shape[0], y2)
            face = img_bgr[y1:y2, x1:x2]

            if face.shape[0] == 0 or face.shape[1] == 0: continue

            # Preprocess the cropped face for the MobileNetV2 classifier
            # This now includes the BGR to RGB conversion
            face_array_processed = preprocess_for_mobilenet(face)
            
            pred = mask_model.predict(face_array_processed, verbose=0)[0]
            label_idx = np.argmax(pred)
            confidence = pred[label_idx]

            label = class_labels_static[label_idx]
            color = (0, 255, 0) if label.lower() == "with mask" else (0, 0, 255)

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            display_text = f"{label} ({confidence*100:.1f}%)"
            font_scale = 0.5 ; font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = x1 ; text_y = y1 - 10
            if y1 - (text_height + 5 + baseline) < 0: text_y = y1 + text_height + 5 + baseline
            cv2.rectangle(img_bgr, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y), color, -1)
            cv2.putText(img_bgr, display_text, (text_x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    else:
        st.warning("No faces detected in the uploaded image.")

    final_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(final_img, caption="Detection Result", use_column_width=True)