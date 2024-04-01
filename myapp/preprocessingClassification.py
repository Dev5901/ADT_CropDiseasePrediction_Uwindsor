#importing packages
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained InceptionV3 model
model = load_model('myapp/inceptionV3_model.h5')  # Provide the path to your saved model

# Load and preprocess the input image
img_path = 'myapp/spilt_Dataset/test/Blight/Corn_Blight (39).jpg'  # Replace with the path to your input image
img = image.load_img(img_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale pixel values to the range [0, 1]

# Make predictions
predictions = model.predict(img_array)

# Get the class label with the highest probability
predicted_class = np.argmax(predictions[0])

# Map class indices to class labels (assuming classes are labeled as 0, 1, 2, 3)
class_labels = {0: 'Blight', 1: 'Common Rust', 2: 'Gray Leaf Spot', 3: 'Healthy'}

# Get the predicted class label
predicted_label = class_labels[predicted_class]

# Print the results
print(f'Predicted class: {predicted_label}')
#print(f'Class probabilities: {predictions[0]}')



#imagePath = 'C:/Users/Dev Prajapati/PycharmProjects/ADTProject/dataset/spilt_Dataset/val/Blight/Corn_Blight (713).jpg'

image = cv2.imread(img_path)
#cv2.imshow('Input image', image)
# converting BGR to HSV
HSV_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#cv2.imshow('HSV IMAGE', HSV_img)


lower_color = np.array([5, 50, 50], dtype="uint8")
upper_color = np.array([15, 255, 255], dtype="uint8")
mask = cv2.inRange(HSV_img, lower_color, upper_color)
#cv2.imshow('Mask 1', mask)

# morphological operations,erosion followed by dilation
kernel = np.ones((4, 4), np.uint8)
erosion = cv2.erode(mask, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)
#cv2.imshow('opening process', dilation)

# gaussian blur
blur = cv2.GaussianBlur(dilation, (5, 5), 0)
#cv2.imshow('Gausssian Blur', blur)

# masking with original image
mask2 = cv2.bitwise_and(image, image, mask=blur)

hsv_image = cv2.cvtColor(mask2, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the brown color in HSV
lower_brown = np.array([5, 50, 50])
upper_brown = np.array([15, 255, 255])

# Threshold the HSV image to get only brown colors
brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

# Calculate the total number of pixels in the image
total_pixels = np.prod(image.shape[:2])

# Calculate the number of brown pixels
brown_pixels = cv2.countNonZero(brown_mask)

# Calculate the brown percentage
brown_percentage = (brown_pixels / total_pixels) * 100

print(f'Severity Percentage: {brown_percentage:.2f}%')

#cv2.waitKey(0)
cv2.destroyAllWindows()