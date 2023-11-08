import numpy as np
import pytesseract
import cv2

# Setting environment variable for pytesseract
pytesseract.pytesseract.tesseract_cmd = r'path to tesseract.exe'

# Opening an image with cv2
filename = 'path to an image from which text has to be extacted'
img = cv2.imread(filename)

# Processing the image to denoise, deblur and sharpen the image
norm_img = np.zeros((img.shape[0], img.shape[1]))
img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)[1]
img = cv2.GaussianBlur(img, (1, 1), 0)

# Displaying the processed image
cv2.imshow("Processed Image",img)
cv2.waitKey(0)

# Generating text from the image using OCR
text = pytesseract.image_to_string(img)

print(text)
