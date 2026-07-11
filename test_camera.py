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

def main():
    parser = argparse.ArgumentParser(description="License Plate Recognition Live Camera Pipeline")
    parser.add_argument(
        "--source", 
        type=str, 
        default="0",
        help="Camera source. Use '0' for default webcam, or an IP Camera URL (e.g., 'http://192.168.1.100:8080/video')"
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

    # Handle integer (webcam index) or string (IP camera URL)
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera source {args.source}")
        return

    print(f"Successfully opened camera source: {args.source}")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. (Make sure the camera stream is active)")
            break

        # Run YOLO detection
        results = yolo_model(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Draw bounding box for the plate
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if conf > 0.5: # Confidence threshold
                    # Crop the detected number plate
                    cropped_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    
                    if cropped_img.size > 0:
                        # Run PaddleOCR on the cropped image
                        ocr_result = ocr.ocr(cropped_img, cls=True)
                        
                        plate_text = ""
                        if ocr_result and ocr_result[0]:
                            for idx, res in enumerate(ocr_result):
                                for line in res:
                                    text = line[1][0]
                                    text_conf = line[1][1]
                                    if text_conf > 0.6:
                                        plate_text += text + " "
                        
                        if plate_text:
                            # Display OCR text on the frame
                            cv2.putText(frame, plate_text.strip(), (x1, max(0, y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            print(f"Detected Plate: {plate_text.strip()} (Conf: {conf:.2f})")

        # Show the frame
        cv2.imshow('ANPR Live Feed', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
