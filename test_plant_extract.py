from cv2 import imread, cvtColor, COLOR_BGR2HSV, inRange, findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, boundingRect, imshow, waitKey, destroyAllWindows, imwrite, contourArea
import numpy as np

# Load the image
image_path = '/path/to/your/image.png'
image = imread(image_path)

# Convert to HSV color space
hsv = cvtColor(image, COLOR_BGR2HSV)

# Define color range for the plant (green color range)
lower_green = np.array([35, 40, 40])  # Adjust based on your plant's color
upper_green = np.array([85, 255, 255])

# Create a mask for the green color (plant)
mask = inRange(hsv, lower_green, upper_green)

# Find contours of the plant regions
contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

# Find the bounding box around the largest contour, assuming it's the plant
if contours:
    largest_contour = max(contours, key=contourArea)
    x, y, w, h = boundingRect(largest_contour)
    
    # Calculate the crop size to center the plant in the middle
    center_x, center_y = x + w // 2, y + h // 2
    crop_size = max(w, h)  # Use the larger dimension for a square crop

    # Calculate crop coordinates, ensuring they stay within image bounds
    x_start = max(center_x - crop_size // 2, 0)
    y_start = max(center_y - crop_size // 2, 0)
    x_end = min(center_x + crop_size // 2, image.shape[1])
    y_end = min(center_y + crop_size // 2, image.shape[0])

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]

    # Display or save the cropped image
    imshow('Cropped Image', cropped_image)
    waitKey(0)
    destroyAllWindows()
    
    # Save the cropped image
    #imwrite('/path/to/save/cropped_plant_image.png', cropped_image)
else:
    print("No plant detected.")
