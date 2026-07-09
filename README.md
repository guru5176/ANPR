# 🚗 Automatic Number Plate Recognition (ANPR) System

A robust, deep-learning-based Automatic Number Plate Recognition (ANPR) pipeline utilizing **YOLO** for license plate detection and **PaddleOCR** for text extraction, optimized for processing challenging and faded license plates.

---

## 📌 Features
- **Object Detection**: Detects license plates dynamically in frames using a custom-trained YOLO model (`model/exp-4.pt`).
- **High-Accuracy OCR**: Extracts license plate text using PaddleOCR with angle classification enabled.
- **Indian License Plate Filtering**: Formats and validates text outputs against standard Indian license plate structures.
- **Image Restoration (Optional)**: Provides custom OpenCV image preprocessing (CLAHE + Gaussian Adaptive Thresholding + Erosion) to restore faded plate details prior to OCR.

---

## 🛠️ Repository Structure

```directory
├── dataset/                     # Directory for training or evaluation datasets
├── model/                       # Trained weights & models
│   ├── exp-4.pt                 # Custom YOLOv8 model for license plate detection
│   └── ch_lprnet_baseline18_deployable.onnx  # Pre-trained LPRNet baseline model
├── testimg/                     # Test images (e.g., UP17.jpg)
├── vc/                          # Visual C++ redistributable packages
├── opencv_restoration.py        # [Optional] Faded plate preprocessing utility
├── test_pipeline.py             # Main end-to-end detection and recognition pipeline
├── test_lprnet.py               # PaddleOCR script with Indian license plate pattern matching
├── test_paddleocr.py            # Basic PaddleOCR test script
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Dependencies
Install the required packages. You will need `torch`, `ultralytics`, `paddlepaddle` (or `paddlepaddle-gpu` for CUDA support), and `paddleocr`.

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install ultralytics paddleocr paddlepaddle opencv-python
```

---

## 💻 Script Reference & Usage

### 1. End-to-End Pipeline (`test_pipeline.py`)
This is the primary script. It detects the license plate using the custom YOLO model, crops the detected plate, and feeds it directly into PaddleOCR.

```bash
python test_pipeline.py --image "testimg/UP17.jpg" --model "model/exp-4.pt"
```

### 2. Plate Verification & Pattern Matching (`test_lprnet.py`)
Applies custom regex rules targeting standard Indian License Plate structures:
$$\text{State Code (2 letters)} + \text{Zone/District Code (2 digits)} + \text{Series (1-3 letters)} + \text{Unique No. (4 digits)}$$

Runs PaddleOCR and filters out junk non-plate detections:
```bash
python test_lprnet.py --image "testimg/UP17.jpg"
```

### 3. Basic OCR Testing (`test_paddleocr.py`)
Performs a direct text extraction test on an image using PaddleOCR:
```bash
python test_paddleocr.py
```

### 4. OpenCV Image Restoration (`opencv_restoration.py`) [OPTIONAL]
An optional utility module designed to pre-process license plate crops before passing them to the OCR engine. 

#### 💡 Use Case
Standard OCR engines like PaddleOCR can struggle to read characters on license plates that are **faded, dusty, weathered, or poorly lit**. This script acts as an intermediate preprocessing step between YOLO detection and OCR extraction to enhance text readability.

**Do not use this by default**; it is designed specifically as an optional enhancement for low-contrast/faded plate images.

#### 🛠️ Key Processing Steps:
1. **Grayscale Conversion**: Strips away color noise to focus purely on text contours.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Maximizes local contrast to reveal faded characters.
3. **Adaptive Gaussian Thresholding**: Evaluates local neighborhoods to force even faintly visible text into 100% solid black characters.
4. **Morphological Erosion**: Thickens the resulting black text strokes so the OCR engine has more solid lines to process.

#### 💻 How to Integrate:
```python
from opencv_restoration import restore_faded_plate

# 1. Get raw plate crop from YOLO
cropped_plate = frame[y1:y2, x1:x2]

# 2. Apply optional restoration for faded/low-contrast plates
restored_plate = restore_faded_plate(cropped_plate)

# 3. Feed the enhanced plate crop into PaddleOCR
ocr_output = ocr.ocr(restored_plate, cls=False)
```

---

## 👥 Authors & Contributors
* **iam_just_ken** ([github/guru5176](https://github.com/guru5176))
