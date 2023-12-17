
import cv2
import numpy as np

# Read the image + convert to greyscale
image = cv2.imread('C:\Users\Owner\Documents\GitHub\AER850_Project3') 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Edge detection using Canny
edges = cv2.Canny(thresholded_image, 50, 150) 

# Contour detection
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours based on area
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]  # Define threshold_area

# Create a mask
mask = np.zeros_like(gray_image)

# Draw contours on the mask
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Use bitwise_and to extract the object
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()