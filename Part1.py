import cv2
import numpy as np

# Step 1: Read the image
image = cv2.imread(r'C:\Users\sshak\Documents\GitHub\AER850_Project3\motherboard_image.JPEG') 

# Step 2: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Thresholding
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Step 4: Edge detection using Canny
edges = cv2.Canny(thresholded_image, 50, 150)  # Adjust parameters as needed

# Step 5: Contour detection
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Filter out small contours based on area
threshold_area = 2  # Adjust this value based on your requirements
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

# Step 7: Create a mask
mask = np.zeros_like(gray_image)

# Step 8: Draw contours on the mask
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Step 9: Use bitwise_and to extract the object
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Step 10: Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()