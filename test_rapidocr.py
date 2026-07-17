import cv2
from rapidocr_onnxruntime import RapidOCR

def test_rapid_ocr(image_path=None):
    ocr = RapidOCR()
    
    if image_path:
        print(f"Testing RapidOCR on image: {image_path}")
        result, elapse = ocr(image_path)
    else:
        print("No image provided. Please provide an image path to test.")
        return
        
    if result:
        for idx, line in enumerate(result):
            print(f"Line {idx}: {line}")
    else:
        print("No text detected.")
        
    print(f"Elapse time: {elapse}")

if __name__ == '__main__':
    # Replace 'test_image.jpg' with a valid image path in your directory
    image_path = r'D:\anpr nvidia\testimg\ANPR.pdf_page_27.png' # Example: 'car_plate.jpg'
    
    import os
    if os.path.exists(image_path):
        test_rapid_ocr(image_path)
    else:
        print(f"RapidOCR test script initialized.")
        print(f"Error: Could not find image at '{image_path}'.")
        print(f"Please update the 'image_path' variable with a valid image path to test.")
