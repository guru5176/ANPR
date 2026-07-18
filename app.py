import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# Suppress warnings
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_mkldnn'] = '0'

import torch
import cv2
import time
import threading
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

# Global variables for models and frame buffers
yolo_model = None
paddle_ocr = None
rapid_ocr = None

latest_rapid_frame = None
latest_paddle_frame = None
lock = threading.Lock()

def init_models():
    global yolo_model, paddle_ocr, rapid_ocr
    if yolo_model is None:
        print("Loading YOLO model...")
        yolo_model = YOLO(r"d:\anpr nvidia\model\exp-4.pt")
    
    if paddle_ocr is None:
        print("Initializing PaddleOCR...")
        try:
            from paddleocr import PaddleOCR
            paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        except Exception as e:
            print(f"Error loading PaddleOCR: {e}")
            paddle_ocr = None
            
    if rapid_ocr is None:
        print("Initializing RapidOCR...")
        try:
            from rapidocr_onnxruntime import RapidOCR
            rapid_ocr = RapidOCR()
        except Exception as e:
            print(f"Error loading RapidOCR: {e}")
            rapid_ocr = None

def video_processing_loop():
    global latest_rapid_frame, latest_paddle_frame
    init_models()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Background video processing thread started successfully.")
    
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue
        
        # Create separate copies for each stream representation
        frame_rapid = frame.copy()
        frame_paddle = frame.copy()

        # Run YOLO detection once
        results = yolo_model(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf > 0.4:
                    # Crop the detected number plate
                    cropped_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    
                    if cropped_img.size > 0:
                        # Scale up the cropped image to help OCR
                        cropped_img_resized = cv2.resize(cropped_img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                        
                        # Draw bounding box for Rapid OCR stream (Cyan)
                        cv2.rectangle(frame_rapid, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        
                        # Draw bounding box for Paddle OCR stream (Green)
                        cv2.rectangle(frame_paddle, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 1. Run RapidOCR
                        if rapid_ocr is not None:
                            try:
                                ocr_result, elapse = rapid_ocr(cropped_img_resized)
                                rapid_text = ""
                                if ocr_result:
                                    for line in ocr_result:
                                        if line and len(line) > 2:
                                            text = line[1]
                                            text_conf = float(line[2])
                                            if text_conf > 0.4:
                                                rapid_text += text + " "
                                rapid_text = rapid_text.strip()
                                if rapid_text:
                                    cv2.putText(frame_rapid, rapid_text, (x1, max(0, y1 - 10)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                            except Exception:
                                pass

                        # 2. Run PaddleOCR
                        if paddle_ocr is not None:
                            try:
                                ocr_result = paddle_ocr.ocr(cropped_img_resized, cls=True)
                                paddle_text = ""
                                if ocr_result and ocr_result[0]:
                                    for idx, res in enumerate(ocr_result):
                                        for line in res:
                                            if line and len(line) > 1:
                                                text = line[1][0]
                                                text_conf = line[1][1]
                                                if text_conf > 0.4:
                                                    paddle_text += text + " "
                                paddle_text = paddle_text.strip()
                                if paddle_text:
                                    cv2.putText(frame_paddle, paddle_text, (x1, max(0, y1 - 10)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            except Exception:
                                pass

        # Encode processed frames to JPEG
        ret_r, buffer_r = cv2.imencode('.jpg', frame_rapid)
        ret_p, buffer_p = cv2.imencode('.jpg', frame_paddle)

        if ret_r and ret_p:
            with lock:
                latest_rapid_frame = buffer_r.tobytes()
                latest_paddle_frame = buffer_p.tobytes()
        
        # Prevent high CPU usage by introducing a tiny sleep
        time.sleep(0.01)

def generate_rapid():
    while True:
        with lock:
            frame_bytes = latest_rapid_frame
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

def generate_paddle():
    while True:
        with lock:
            frame_bytes = latest_paddle_frame
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Dual ANPR Control Center</title>
                <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
                <style>
                    :root {
                        --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                        --glass-bg: rgba(255, 255, 255, 0.03);
                        --glass-border: rgba(255, 255, 255, 0.08);
                        --accent-cyan: #06b6d4;
                        --accent-green: #10b981;
                    }
                    
                    * {
                        box-sizing: border-box;
                        margin: 0;
                        padding: 0;
                    }
                    
                    body {
                        background: var(--bg-gradient);
                        min-height: 100vh;
                        color: #f8fafc;
                        font-family: 'Outfit', sans-serif;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        padding: 20px;
                        overflow-x: hidden;
                    }
                    
                    .dashboard {
                        background: var(--glass-bg);
                        backdrop-filter: blur(16px);
                        -webkit-backdrop-filter: blur(16px);
                        border: 1px solid var(--glass-border);
                        border-radius: 24px;
                        padding: 30px;
                        max-width: 1200px;
                        width: 100%;
                        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
                        text-align: center;
                    }
                    
                    h1 {
                        font-size: 2.5rem;
                        font-weight: 700;
                        background: linear-gradient(to right, #22d3ee, #818cf8, #34d399);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        margin-bottom: 8px;
                    }
                    
                    p.subtitle {
                        color: #94a3b8;
                        font-size: 1.05rem;
                        margin-bottom: 24px;
                    }
                    
                    .streams-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
                        gap: 24px;
                        margin-top: 10px;
                    }
                    
                    @media (max-width: 600px) {
                        .streams-grid {
                            grid-template-columns: 1fr;
                        }
                    }
                    
                    .feed-card {
                        background: rgba(0, 0, 0, 0.2);
                        border: 1px solid var(--glass-border);
                        border-radius: 16px;
                        padding: 16px;
                        display: flex;
                        flex-direction: column;
                        gap: 12px;
                    }
                    
                    .feed-card.rapid-card {
                        border-top: 4px solid var(--accent-cyan);
                    }
                    
                    .feed-card.paddle-card {
                        border-top: 4px solid var(--accent-green);
                    }
                    
                    .feed-card h3 {
                        font-size: 1.25rem;
                        font-weight: 600;
                    }
                    
                    .feed-card.rapid-card h3 {
                        color: var(--accent-cyan);
                    }
                    
                    .feed-card.paddle-card h3 {
                        color: var(--accent-green);
                    }
                    
                    .video-container {
                        position: relative;
                        border-radius: 12px;
                        overflow: hidden;
                        border: 1px solid var(--glass-border);
                        background: #000;
                        aspect-ratio: 4/3;
                    }
                    
                    .video-container img {
                        width: 100%;
                        height: 100%;
                        object-fit: contain;
                    }
                    
                    .status-indicator {
                        display: inline-flex;
                        align-items: center;
                        gap: 8px;
                        font-size: 0.9rem;
                        color: #cbd5e1;
                        background: rgba(255, 255, 255, 0.02);
                        padding: 8px 16px;
                        border-radius: 50px;
                        border: 1px solid var(--glass-border);
                        margin-bottom: 20px;
                    }
                    
                    .pulse-dot {
                        width: 8px;
                        height: 8px;
                        background-color: var(--accent-green);
                        border-radius: 50%;
                        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
                        animation: pulse 1.6s infinite;
                    }
                    
                    @keyframes pulse {
                        0% {
                            transform: scale(0.95);
                            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
                        }
                        70% {
                            transform: scale(1);
                            box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
                        }
                        100% {
                            transform: scale(0.95);
                            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
                        }
                    }
                </style>
            </head>
            <body>
                <div class="dashboard">
                    <h1>Dual ANPR Comparison Center</h1>
                    <p class="subtitle">Real-time side-by-side license plate detection feed</p>
                    
                    <div class="status-indicator">
                        <span class="pulse-dot"></span>
                        <span>Dual Streams Active</span>
                    </div>
                    
                    <div class="streams-grid">
                        <div class="feed-card rapid-card">
                            <h3>Rapid OCR Engine</h3>
                            <div class="video-container">
                                <img src="/video_feed/rapid" alt="Rapid OCR Stream">
                            </div>
                        </div>
                        <div class="feed-card paddle-card">
                            <h3>Paddle OCR Engine</h3>
                            <div class="video-container">
                                <img src="/video_feed/paddle" alt="Paddle OCR Stream">
                            </div>
                        </div>
                    </div>
                </div>
            </body>
        </html>
    ''')

@app.route('/video_feed/rapid')
def video_feed_rapid():
    return Response(generate_rapid(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/paddle')
def video_feed_paddle():
    return Response(generate_paddle(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the background frame processing thread
    threading.Thread(target=video_processing_loop, daemon=True).start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000)
