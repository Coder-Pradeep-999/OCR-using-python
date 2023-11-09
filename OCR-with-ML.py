import numpy as np
import pytesseract
import cv2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Setting environment variable for pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Opening an image with cv2
filename = 'path to an image'
img = cv2.imread(filename)


# Define a function to extract features from an image
# Feature engineering for image processing optimization
def extract_features(image_path):
    img = cv2.imread(image_path)
    mean_pixel_values = np.mean(img, axis=(0, 1))
    std_pixel_values = np.std(img, axis=(0, 1))
    aspect_ratio = img.shape[1] / img.shape[0]
    return np.concatenate([mean_pixel_values, std_pixel_values, [aspect_ratio]])

# Load multiple images from a directory
image_dir = 'path to the directory containing images to train the model'
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')]

# Create a list to store features and targets
features_list = []
targets_list = []

# Extract features and targets from each image
for image_file in image_files:
    features = extract_features(image_file)
    # Dummy target for demonstration purposes, replace with your actual target
    target = 0.5
    features_list.append(features)
    targets_list.append(target)


# Convert lists to NumPy arrays
features_array = np.array(features_list)
targets_array = np.array(targets_list)

# Train-test split with multiple samples
X_train, X_test, y_train, y_test = train_test_split(features_array, targets_array, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Use the trained model to predict performance improvement
performance_improvement = model.predict(features_array)
print(f'Predicted Performance Improvement: {performance_improvement[0]}')


# Processing the image to denoise, deblur, and sharpen the image
# You can use the predicted performance improvement to adjust parameters
# For simplicity, I'm using a fixed threshold and GaussianBlur in this example
threshold_value = 100 - performance_improvement[0] * 50
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_TOZERO)[1]
img = cv2.GaussianBlur(img, (1, 1), 0)

# Displaying the processed image
cv2.imshow("Processed Image", img)
cv2.waitKey(0)

# Generating text from the image using OCR
text = pytesseract.image_to_string(img)

print(text)
