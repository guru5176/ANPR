import cv2
import numpy as np

def restore_faded_plate(cropped_plate):
    """
    Takes the raw YOLO crop of a faded plate and mathematically 
    restores the text contrast for the OCR engine.
    """
    # 1. Strip away all color (Grayscale)
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This aggressively pulls out faint details by fixing the local contrast.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    high_contrast = clahe.apply(gray)
    
    # 3. Adaptive Thresholding
    # Instead of looking at the whole image, it looks at tiny 11x11 pixel grids.
    # If the faded 'Y' is even 5% darker than the white background next to it, 
    # this forces it to become 100% solid black.
    binary = cv2.adaptiveThreshold(
        high_contrast, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 4. Morphological "Thickening" (Erosion)
    # Because the faded letters are now black on a white background, 
    # we "erode" the white pixels, which effectively thickens the black text lines.
    kernel = np.ones((2, 2), np.uint8)
    restored_image = cv2.erode(binary, kernel, iterations=1)
    
    # Optional: Convert back to 3-channel (BGR) if PaddleOCR expects a color format
    # restored_image = cv2.cvtColor(restored_image, cv2.COLOR_GRAY2BGR)
    
    return restored_image

# --- How to use it in your pipeline ---
# 1. cropped_plate = frame[y1:y2, x1:x2]  <-- From YOLO
# 2. clean_plate = restore_faded_plate(cropped_plate)
# 3. ocr_output = self.ocr.ocr(clean_plate, cls=False)
