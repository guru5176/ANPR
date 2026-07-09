import os
import torch
import cv2
import time
from flask import Flask, Response, render_template_string
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Suppress PaddlePaddle warnings for cleaner output
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_mkldnn'] = '0'

app = Flask(__name__)

# Initialize models globally
yolo_model = None
ocr = None

def init_models():
    global yolo_model, ocr
    if yolo_model is None:
        print("Loading YOLO model...")
        yolo_model = YOLO(r"d:\anpr nvidia\model\exp-4.pt")
    if ocr is None:
        print("Initializing PaddleOCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en')

def generate_frames():
    init_models()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open laptop webcam (source 0).")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Run YOLO detection
            results = yolo_model(frame, verbose=False)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Draw bounding box for the plate
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if conf > 0.4:
                        # Crop the detected number plate
                        cropped_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        
                        if cropped_img.size > 0:
                            # Optional: scale up the cropped image to help OCR
                            cropped_img = cv2.resize(cropped_img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                            
                            # Run PaddleOCR on the cropped image
                            ocr_result = ocr.ocr(cropped_img, cls=True)
                            
                            plate_text = ""
                            if ocr_result and ocr_result[0]:
                                for idx, res in enumerate(ocr_result):
                                    for line in res:
                                        if line and len(line) > 1:
                                            text = line[1][0]
                                            text_conf = line[1][1]
                                            if text_conf > 0.4:
                                                plate_text += text + " "
                            
                            plate_text = plate_text.strip()
                            
                            if plate_text:
                                cv2.putText(frame, plate_text, (x1, max(0, y1 - 10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame for multipart HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <html>
            <head>
                <title>ANPR Live Camera</title>
                <style>
                    body { background-color: #121212; color: white; text-align: center; font-family: sans-serif; }
                    img { border: 2px solid #4CAF50; border-radius: 8px; margin-top: 20px; max-width: 100%; }
                </style>
            </head>
            <body>
                <h1>ANPR Live Web Feed</h1>
                <p>Hold up a license plate to your laptop camera!</p>
                <img src="/video_feed" />
            </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start Flask server
    app.run(host='0.0.0.0', port=5000)
