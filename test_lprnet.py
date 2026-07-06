import os
import argparse
import re
from paddleocr import PaddleOCR

# Suppress PaddlePaddle warnings for cleaner output
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_mkldnn'] = '0'

def filter_plate_text(raw_text):
    """
    Filters raw OCR text to extract valid Indian license plates, 
    accounting for missing leading zeros and varied alphabet series.
    """
    flexible_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$'
    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    
    if re.match(flexible_pattern, clean_text):
        return clean_text
    return None


def process_image(img_path, ocr):
    print(f"\nAnalyzing image: {img_path}")
    if not os.path.exists(img_path):
        print("Error: Image not found!")
        return
        
    result = ocr.ocr(img_path, cls=True)
    
    if result and result[0]:
        for idx, res in enumerate(result):
            for line in res:
                # line format: [[box coords], (text, confidence)]
                box = line[0]
                text = line[1][0]
                conf = line[1][1]
                
                filtered_text = filter_plate_text(text)
                
                if filtered_text:
                    print(f"[ACCEPTED] Detected Plate: '{filtered_text}' (Raw: '{text}', Confidence: {conf:.4f})")
                else:
                    print(f"[IGNORED]  Junk Text: '{text}' (Confidence: {conf:.4f})")
    else:
        print("No text detected.")

def main():
    parser = argparse.ArgumentParser(description="License Plate Recognition using PaddleOCR")
    parser.add_argument(
        "--image", 
        type=str, 
        default=r"d:\anpr nvidia\testimg\UP17.jpg",
        help="Path to the license plate image"
    )
    args = parser.parse_args()

    print("Initializing PaddleOCR...")
    # use_angle_cls=True allows detecting text at different orientations
    # lang='en' for English characters
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Process the provided image
    process_image(args.image, ocr)

if __name__ == '__main__':
    main()
