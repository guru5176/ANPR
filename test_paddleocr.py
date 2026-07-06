from paddleocr import PaddleOCR
import cv2

def main():
    img_path = r"d:\anpr nvidia\testimg\20260530_194802.jpg"
    
    # Initialize PaddleOCR model. 
    # use_angle_cls=True allows detecting text at different orientations
    # lang='en' for English characters usually found in license plates
    print(f"Initializing PaddleOCR and analyzing image: {img_path}")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Run OCR on the image
    result = ocr.ocr(img_path, cls=True)
    
    print("\n--- PaddleOCR Results ---")
    if result and result[0]:
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                # line format: [[box coords], (text, confidence)]
                box = line[0]
                text = line[1][0]
                conf = line[1][1]
                print(f"Detected Text: '{text}' (Confidence: {conf:.4f})")
    else:
        print("No text detected.")

if __name__ == '__main__':
    main()
