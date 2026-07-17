import os
import torch
import cv2
import time
import re

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_mkldnn'] = '0'

from paddleocr import PaddleOCR
from ultralytics import YOLO

def filter_plate_text(raw_text):
    flexible_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$'
    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    if re.match(flexible_pattern, clean_text):
        return clean_text
    return None

def main():
    video_path = r"d:\anpr nvidia\testimg\WhatsApp Video 2026-07-11 at 8.24.56 PM.mp4"
    model_path = r"d:\anpr nvidia\model\exp-4.pt"
    
    print(f"Loading YOLO model from {model_path}...")
    yolo_model = YOLO(model_path)
    
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {video_path} ({total_frames} frames total)")
    
    frame_count = 0
    detected_plates = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # Log progress every 50 frames
        if frame_count % 50 == 0 or frame_count == 1:
            print(f"Processing frame {frame_count}/{total_frames}...")
            
        # Run YOLO detection
        results = yolo_model(frame, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf > 0.25:
                    cropped_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if cropped_img.size > 0:
                        cropped_img = cv2.resize(cropped_img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                        ocr_result = ocr.ocr(cropped_img, cls=True)
                        
                        plate_text = ""
                        if ocr_result and ocr_result[0]:
                            for idx, res in enumerate(ocr_result):
                                for line in res:
                                    if line and len(line) > 1:
                                        text = line[1][0]
                                        text_conf = line[1][1]
                                        if text_conf > 0.3:
                                            plate_text += text + " "
                                            
                        plate_text = plate_text.strip()
                        if plate_text:
                            filtered = filter_plate_text(plate_text)
                            detected_plates.append({
                                'frame': frame_count,
                                'raw': plate_text,
                                'filtered': filtered,
                                'confidence': conf
                            })
                            print(f"Frame {frame_count}: Raw OCR: '{plate_text}' | Filtered: '{filtered}' (YOLO Conf: {conf:.2f})")
                            
    cap.release()
    
    print("\n--- Processing Finished ---")
    print(f"Total Frames Analyzed: {frame_count}")
    print("\nAll detections:")
    for det in detected_plates:
        print(f"Frame {det['frame']}: Raw: '{det['raw']}' | Filtered: '{det['filtered']}' | Conf: {det['confidence']:.2f}")

if __name__ == '__main__':
    main()
