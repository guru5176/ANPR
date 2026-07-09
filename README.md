# ANPR

## Optional Files

* **`opencv_restoration.py`**: An optional script containing the `restore_faded_plate` function. This function uses OpenCV (Adaptive Thresholding and Morphological Filtering) to mathematically restore the text contrast of faded license plates for better OCR extraction. It is intended to sit right between the YOLO crop and the PaddleOCR extraction steps.
