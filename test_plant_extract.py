from cv2 import imread, imshow, waitKey, destroyAllWindows
from numpy import ndarray

from classes.image_lib import ImageAgent
from classes.util_lib import Rect

# # Load the image
# image_path = '/path/to/your/image.png'
# image = imread(image_path)

# # Convert to HSV color space
# hsv = cvtColor(image, COLOR_BGR2HSV)

# # Define color range for the plant (green color range)
# lower_green = np.array([35, 40, 40])  # Adjust based on your plant's color
# upper_green = np.array([85, 255, 255])

# # Create a mask for the green color (plant)
# mask = inRange(hsv, lower_green, upper_green)

# # Find contours of the plant regions
# contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

# # Find the bounding box around the largest contour, assuming it's the plant
# if contours:
#     largest_contour = max(contours, key=contourArea)
#     x, y, w, h = boundingRect(largest_contour)
    
#     # Calculate the crop size to center the plant in the middle
#     center_x, center_y = x + w // 2, y + h // 2
#     crop_size = max(w, h)  # Use the larger dimension for a square crop

#     # Calculate crop coordinates, ensuring they stay within image bounds
#     x_start = max(center_x - crop_size // 2, 0)
#     y_start = max(center_y - crop_size // 2, 0)
#     x_end = min(center_x + crop_size // 2, image.shape[1])
#     y_end = min(center_y + crop_size // 2, image.shape[0])

#     # Crop the image
#     cropped_image = image[y_start:y_end, x_start:x_end]

#     # Display or save the cropped image
#     imshow('Cropped Image', cropped_image)
#     waitKey(0)
#     destroyAllWindows()
    
#     # Save the cropped image
#     #imwrite('/path/to/save/cropped_plant_image.png', cropped_image)
# else:
#     print("No plant detected.")

def main():
    image_path : str = "bin/20240527_week7/top_20240527_week7/0000039.png"
    image_agent : ImageAgent = ImageAgent()
    image : ndarray = image_agent.LoadImage(image_path, image_agent.ColorModeEnum.rgb_)

    print(image.shape)

    mask : ndarray = image_agent.FindPlantMask(image)
    
    plant_contours : list[Rect[int]] | None = image_agent.FindPlantContour(mask)

    if plant_contours is None:
        print("No plant detected.")

    for plant_contour in plant_contours:
        center_x, center_y = plant_contour.point_.x_ + plant_contour.size_.width_ // 2, plant_contour.point_.y_ + plant_contour.size_.height_ // 2
        crop_size = max(plant_contour.size_.width_, plant_contour.size_.height_)

        x_start = max(center_x - crop_size // 2, 0)
        y_start = max(center_y - crop_size // 2, 0)
        x_end = min(center_x + crop_size // 2, image.shape[1])
        y_end = min(center_y + crop_size // 2, image.shape[0])

        cropped_image = image[y_start:y_end, x_start:x_end]

        imshow('Cropped Image', cropped_image)
        waitKey(0)
        destroyAllWindows()


if __name__  == "__main__":
    main()