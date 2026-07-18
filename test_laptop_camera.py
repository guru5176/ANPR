import os
import torch
import argparse
import cv2
import time
from rapidocr_onnxruntime import RapidOCR
from ultralytics import YOLO

# Suppress warnings
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_mkldnn'] = '0'

def main():
    parser = argparse.ArgumentParser(description="Laptop Camera Live Pipeline for ANPR")
    parser.add_argument(
        "--model", 
        type=str, 
        default=r"d:\anpr nvidia\model\exp-4.pt",
        help="Path to the YOLO model"
    )
    args = parser.parse_args()

    print(f"Loading YOLO model from {args.model}...")
    yolo_model = YOLO(args.model).to('cuda')

    print("Initializing RapidOCR...")
    ocr = RapidOCR()

    # Source 0 is typically the built-in laptop webcam
    source = 0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open laptop webcam (source 0).")
        return

    print("Successfully opened laptop webcam!")
    print("Hold up images of number plates to the camera to see live detections.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check your webcam.")
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
                
                if conf > 0.4: # Decent confidence threshold for live webcam
                    # Crop the detected number plate
                    cropped_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    
                    if cropped_img.size > 0:
                        # Optional: scale up the cropped image to help OCR
                        cropped_img = cv2.resize(cropped_img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                        
                        # Run RapidOCR on the cropped image
                        ocr_result, elapse = ocr(cropped_img)
                        
                        plate_text = ""
                        if ocr_result:
                            for line in ocr_result:
                                if line and len(line) > 2:
                                    text = line[1]
                                    text_conf = float(line[2])
                                    if text_conf > 0.4:
                                        plate_text += text + " "
                        
                        plate_text = plate_text.strip()
                        
                        if plate_text:
                            # Display OCR text on the frame
                            cv2.putText(frame, plate_text, (x1, max(0, y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            print(f"Live Plate Detected: {plate_text} (Conf: {conf:.2f})")

        # Show the frame
        cv2.imshow('Laptop Camera Live Feed', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
