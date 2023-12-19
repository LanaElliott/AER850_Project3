import cv2
import numpy as np

# Step 1: Read the image
image = cv2.imread(r'C:\Users\sshak\Documents\GitHub\AER850_Project3\motherboard_image.JPEG')

# Step 2: Convert the image to grayscale and apply blur
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (45, 45), 4)

# Step 3: Thresholding
thresholded_image = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 5)

# Step 4: Edge detection using Canny
edges = cv2.Canny(thresholded_image, 1, 1)  
edges = cv2.dilate(edges,None, iterations = 8)

# Step 5: Contour detection
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Filter out contours based on area
min_area_threshold = 10 
max_area_threshold = 5000 

filtered_contours = [cnt for cnt in contours if min_area_threshold < cv2.contourArea(cnt) < max_area_threshold]

# Step 7: Create a mask
mask = np.zeros_like(blur_image)

# Step 8: Draw contours on the mask
cv2.drawContours(mask, contours=[max(contours, key = cv2.contourArea)], contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)

# Step 9: Use bitwise_and to extract the object
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Set window names
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Masked Image', cv2.WINDOW_NORMAL)

# Resize windows
cv2.resizeWindow('Original Image', 800, 600)
cv2.resizeWindow('Masked Image', 800, 600)

# Step 10: Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()