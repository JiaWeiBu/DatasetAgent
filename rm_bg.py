from os import listdir, makedirs
from os.path import exists
from cv2 import imread, fillPoly, bitwise_and, imwrite
from numpy import ndarray, zeros, uint8, array, float32, int32

# i want to load yolo segmentation dataset remove the background and also create a mask for black and white image

input_root : str = "data-test"
image_folder_name : str = "images"
label_folder_name : str = "labels"
input_source : list[str] = ["test", "train", "valid"]
output_root : str = "bg_bin"
mask_dir_name : str = "mask"
bgrm_dir_name : str = "bgrm"

def CheckDir(dir_path : str) -> None:
    if not exists(dir_path):
        print("not found")
        makedirs(dir_path)

def yolo_to_mask(image_path : str, label_path : str) -> ndarray:

    # Read image
    image : ndarray = imread(image_path)
    h, w, _ = image.shape # height, width, channel

    # Create a blank mask
    mask : ndarray = zeros((h, w), dtype=uint8)
    mask2 : ndarray = zeros((h, w), dtype=uint8)

    # Read Yolo Segementation label
    with open(label_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        data = line.strip().split(" ")
        class_id = int(data[0])
        polygon_points = array(data[1:], dtype=float32).reshape(-1, 2)

        # Convert normalized coordinates to pixel coordinates
        polygon_points[:, 0] *= w # Scale x-coordinates
        polygon_points[:, 1] *= h # Scale y-coordinates
        polygon_points = polygon_points.astype(int32) # Convert to integers

        # There are bounding boxes mixed in the label file, draw them as white regions
        if len(polygon_points) < 3:
            continue
        else:
            # Fill the polygon
            fillPoly(img=mask, pts=[polygon_points], color=(255, 0, 0)) # White region for plant

    return mask

def process_image(image_path : str, label_path : str, output_image_path : str, output_mask_path : str) -> None:
    # Generate the segmentation mask
    mask : ndarray = yolo_to_mask(image_path, label_path)

    # Read the image
    image : ndarray = imread(image_path)

    # Apply the mask to remove the background
    result : ndarray = bitwise_and(image, image, mask=mask)

    # Save the result
    imwrite(output_image_path, result)
    imwrite(output_mask_path, mask)

def main() -> None:
    # image_path : str = image_dir + "/" + img_name
    # label_path : str = label_dir + "/" + label_name
    # output_image_path : str = output_dir + "/" + bgrm_dir_name + "/" + img_name
    # output_mask_path : str = output_dir + "/" + mask_dir_name + "/" + img_name

    # Check paths
    CheckDir(output_root + "/" + bgrm_dir_name)
    CheckDir(output_root + "/" + mask_dir_name)

    for source in input_source:
        image_dir : str = input_root + "/" + source + "/" + image_folder_name
        label_dir : str = input_root + "/" + source + "/" + label_folder_name
        output_image_dir : str = output_root + "/" + bgrm_dir_name
        output_mask_dir : str = output_root + "/" + mask_dir_name

        for img_name in listdir(image_dir):
            image_path : str = image_dir + "/" + img_name
            label_path : str = label_dir + "/" + img_name.rstrip(".jpg") + ".txt"
            output_image_path : str = output_image_dir + "/" + img_name
            output_mask_path : str = output_mask_dir + "/" + img_name

            process_image(image_path, label_path, output_image_path, output_mask_path)
        
if __name__ == "__main__":
    main()