# 😷 Real-time Face Mask Detection System
Demo - https://drive.google.com/file/d/1K6XVEwkvJCPtvzUNDXr_PvtBt7pLhwu0/view?usp=sharing
A comprehensive computer vision application that detects faces in real-time and classifies whether individuals are wearing masks or not. Built with Streamlit, YOLOv11, and MobileNetV2, this system provides both live webcam detection and static image analysis capabilities.

## 🚀 Features

- **Real-time Detection**: Live webcam feed with instant mask detection
- **Static Image Analysis**: Upload and analyze individual images
- **High Accuracy**: Fine-tuned MobileNetV2 model for reliable classification
- **User-friendly Interface**: Clean Streamlit web interface
- **Multi-face Support**: Detects and classifies multiple faces simultaneously
- **Confidence Scoring**: Displays prediction confidence percentages
- **Visual Feedback**: Color-coded bounding boxes (Green = With Mask, Red = No Mask)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Computer Vision**: OpenCV, YOLOv11 (face detection)
- **Deep Learning**: TensorFlow/Keras, MobileNetV2 (mask classification)
- **Real-time Processing**: Streamlit-WebRTC
- **Model Hosting**: Hugging Face Hub

## 📁 Project Structure

```
mask-detection-final/
├── app.py                          # Main Streamlit application
├── eda.ipynb                       # Training notebook with model development
├── requirements.txt                # Python dependencies
├── mask_detector_model.h5          # Final trained model
├── mask_classifier_phase1_best.h5  # Phase 1 training checkpoint
├── data/                           # Training dataset
│   ├── with_mask/                  # Images of people wearing masks
│   └── without_mask/               # Images of people without masks
└── venv/                          # Virtual environment
```

## 🎯 Model Architecture

### Face Detection
- **Model**: YOLOv11 (from Hugging Face Hub)
- **Purpose**: Detect and localize faces in images/video frames
- **Output**: Bounding box coordinates for each detected face

### Mask Classification
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Fine-tuning Strategy**: Two-phase training approach
  - **Phase 1**: Freeze base model, train classification head
  - **Phase 2**: Unfreeze base model, fine-tune entire network
- **Input Size**: 224x224 pixels
- **Classes**: 2 (With Mask, Without Mask)
- **Output**: Softmax probabilities for each class

## 📊 Dataset

- **Total Images**: 7,553 (6,043 training + 1,510 validation)
- **Classes**: 
  - With Mask: ~3,000 images
  - Without Mask: ~4,500 images
- **Split**: 80% training, 20% validation
- **Augmentation**: Rotation, zoom, shift, shear, horizontal flip

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Webcam (for real-time detection)
- Sufficient RAM (4GB+ recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mask-detection-final
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser and go to `http://localhost:8501`
   - The app will automatically download the YOLOv11 model on first run

### Using the Application

#### Live Webcam Detection
1. Click "START" to activate your webcam
2. Position yourself in front of the camera
3. The system will detect faces and classify mask usage in real-time
4. Green boxes indicate "With Mask", red boxes indicate "No Mask"
5. Confidence percentages are displayed above each bounding box

#### Static Image Analysis
1. Scroll down to the "Static Image Upload" section
2. Click "Browse files" and select an image (JPG, JPEG, PNG)
3. The system will process the image and display results
4. Multiple faces in the image will be detected and classified

## 🔧 Model Training

The training process is documented in `eda.ipynb` and follows a two-phase approach:

### Phase 1: Head Training
- Freeze MobileNetV2 base model
- Train only the classification head
- Learning rate: 1e-3
- Epochs: 15
- Early stopping with patience: 5

### Phase 2: Fine-tuning
- Unfreeze base model layers
- Fine-tune entire network
- Learning rate: 1e-5
- Epochs: 50
- Learning rate reduction on plateau
- Early stopping with patience: 10

### Training Configuration
- **Image Size**: 224x224
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Data Augmentation**: Enabled

## 📈 Performance

- **Validation Accuracy**: ~97% (Phase 1)
- **Model Size**: ~9MB (MobileNetV2)
- **Inference Speed**: Real-time (30+ FPS on modern hardware)
- **Face Detection Confidence**: 0.4 (configurable)

## 🔍 Technical Details

### Preprocessing Pipeline
1. **Face Detection**: YOLOv11 detects faces with confidence > 0.4
2. **Face Cropping**: Extract face regions using bounding boxes
3. **Color Conversion**: BGR to RGB (MobileNetV2 requirement)
4. **Resizing**: Scale to 224x224 pixels
5. **Normalization**: Divide by 255.0
6. **Classification**: MobileNetV2 predicts mask status

### Key Implementation Notes
- **Color Space Handling**: Critical BGR to RGB conversion for MobileNetV2
- **Class Mapping**: `{'with_mask': 0, 'without_mask': 1}`
- **Model Caching**: Uses Streamlit's `@st.cache_resource` for efficient loading
- **Async Processing**: WebRTC enables smooth real-time video processing

## 🐛 Troubleshooting

### Common Issues

1. **Webcam not working**
   - Ensure camera permissions are granted
   - Check if another application is using the camera
   - Try refreshing the browser page

2. **Model download fails**
   - Check internet connection
   - Verify Hugging Face Hub access
   - Clear browser cache and retry

3. **Low detection accuracy**
   - Ensure good lighting conditions
   - Position face clearly in camera view
   - Check if face is not obstructed

4. **Performance issues**
   - Close other applications using GPU/CPU
   - Reduce webcam resolution if needed
   - Ensure sufficient RAM is available

## 🙏 Acknowledgments

- **YOLOv11**: AdamCodd for the face detection model
- **MobileNetV2**: Google Research for the base architecture
- **Streamlit**: For the web framework
- **OpenCV**: For computer vision capabilities
 
