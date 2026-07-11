import os
import torch
import argparse
import cv2
import time
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Suppress PaddlePaddle warnings for cleaner output
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_mkldnn'] = '0'

def process_image(img_path, yolo_model, ocr):
    print(f"\nAnalyzing image: {img_path}")
    if not os.path.exists(img_path):
        print("Error: Image not found!")
        return

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # Run YOLO detection
    start_yolo = time.time()
    results = yolo_model(img)
    end_yolo = time.time()
    print(f"YOLO detection time: {end_yolo - start_yolo:.4f} seconds")
    
    detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            print(f"Detected number plate with confidence {conf:.4f} at [{x1}, {y1}, {x2}, {y2}]")
            
            # Crop the detected number plate
            cropped_img = img[y1:y2, x1:x2]
            
            # Run PaddleOCR on the cropped image
            start_ocr = time.time()
            ocr_result = ocr.ocr(cropped_img, cls=True)
            end_ocr = time.time()
            print(f"OCR time: {end_ocr - start_ocr:.4f} seconds")
            
            if ocr_result and ocr_result[0]:
                for idx, res in enumerate(ocr_result):
                    for line in res:
                        # line format: [[box coords], (text, confidence)]
                        text = line[1][0]
                        text_conf = line[1][1]
                        print(f"  -> Extracted Text: '{text}' (OCR Confidence: {text_conf:.4f})")
                        detected = True
            else:
                print("  -> No text detected in this plate by PaddleOCR.")
                
    if not detected:
        print("No valid text extracted from the image.")

def main():
    parser = argparse.ArgumentParser(description="License Plate Recognition Pipeline")
    parser.add_argument(
        "--image", 
        type=str, 
        default=r"d:\anpr nvidia\testimg\UP17.jpg",
        help="Path to the test image"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=r"d:\anpr nvidia\model\exp-4.pt",
        help="Path to the YOLO model"
    )
    args = parser.parse_args()

    print(f"Loading YOLO model from {args.model}...")
    yolo_model = YOLO(args.model)

    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Process the provided image
    process_image(args.image, yolo_model, ocr)

if __name__ == '__main__':
    main()
