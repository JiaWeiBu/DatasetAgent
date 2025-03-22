import re
from os import listdir, makedirs
from os.path import exists, join
from shutil import rmtree
from cv2 import imread, imwrite, boundingRect, fillPoly
from numpy import ndarray, zeros, uint8, array, float32, int32

# Paths
input_root: str = "data-test2"
image_folder_name: str = "images"
label_folder_name: str = "labels"
input_source: list[str] = ["test", "train", "valid"]
output_root: str = "processed"
mask_dir_name: str = "mask"
cropped_dir_name: str = "cropped"

# Regex Patterns to Extract Week Number
pattern1 = re.compile(r"(?:week|Week)?(\d+)_60degrees_(\d+)_\w+\.\w+\.[a-z0-9]+\.(jpg|png)", re.IGNORECASE)
pattern2 = re.compile(r"60degree_\d+_(?:week|Week)(\d+)_(\d+)_\w+\.\w+\.[a-z0-9]+\.(jpg|png)", re.IGNORECASE)

def CheckDir(dir_path: str) -> None:
    if not exists(dir_path):
        makedirs(dir_path)

def CleanDir(dir_path: str) -> None:
    if exists(dir_path):
        rmtree(dir_path)
    makedirs(dir_path)

def extract_week(image_name: str):
    match1 = pattern1.match(image_name)
    match2 = pattern2.match(image_name)
    
    if match1:
        return int(match1.group(1))  # Extracted week number
    elif match2:
        return int(match2.group(1))  # Extracted week number
    return None  # No match found

def yolo_to_objects(image_path: str, label_path: str):
    # Read image
    image: ndarray = imread(image_path)
    h, w, _ = image.shape  # height, width, channel
    
    objects = []  # List to store valid objects (bounding box and mask)
    max_width, max_height = 0, 0  # Track maximum object size
    
    # Read YOLO Segmentation labels
    with open(label_path, "r") as file:
        lines = file.readlines()
    
    for line in lines:
        data = line.strip().split(" ")
        polygon_points = array(data[1:], dtype=float32).reshape(-1, 2)
        
        if len(polygon_points) < 3:
            continue  # Ignore invalid objects
        
        # Convert normalized coordinates to pixel coordinates
        polygon_points[:, 0] *= w
        polygon_points[:, 1] *= h
        polygon_points = polygon_points.astype(int32)
        
        # Create mask
        mask = zeros((h, w), dtype=uint8)
        fillPoly(mask, [polygon_points], 255)
        
        # Get bounding box
        x, y, width, height = boundingRect(polygon_points)
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        
        objects.append((x, y, width, height, mask))
    
    return objects, max_width, max_height

def process_images():
    # Clean processed folders
    CleanDir(output_root)
    CleanDir(join(output_root, cropped_dir_name))
    CleanDir(join(output_root, mask_dir_name))
    
    week_max_size = {}  # Dictionary to store max width/height per week
    
    # First pass: Determine maximum crop size per week
    for source in input_source:
        image_dir = join(input_root, source, image_folder_name)
        label_dir = join(input_root, source, label_folder_name)
        
        for img_name in listdir(image_dir):
            week_num = extract_week(img_name)
            if week_num is None:
                continue  # Skip if no week number found
            
            image_path = join(image_dir, img_name)
            label_path = join(label_dir, img_name.rsplit('.', 1)[0] + ".txt")
            
            if not exists(label_path):
                continue  # Skip images without labels
            
            objects, max_width, max_height = yolo_to_objects(image_path, label_path)
            
            if objects:
                if week_num not in week_max_size:
                    week_max_size[week_num] = (max_width, max_height)
                else:
                    prev_max_w, prev_max_h = week_max_size[week_num]
                    week_max_size[week_num] = (max(prev_max_w, max_width), max(prev_max_h, max_height))

    # Second pass: Crop and save objects based on week's max size
    for source in input_source:
        image_dir = join(input_root, source, image_folder_name)
        label_dir = join(input_root, source, label_folder_name)
        
        for img_name in listdir(image_dir):
            week_num = extract_week(img_name)
            if week_num is None or week_num not in week_max_size:
                continue  # Skip if no valid week number
            
            image_path = join(image_dir, img_name)
            label_path = join(label_dir, img_name.rsplit('.', 1)[0] + ".txt")
            
            if not exists(label_path):
                continue
            
            objects, _, _ = yolo_to_objects(image_path, label_path)
            image = imread(image_path)
            
            max_width, max_height = week_max_size[week_num]  # Get max crop size for this week
            
            obj_count = 1
            for x, y, width, height, mask in objects:
                # Ensure uniform crop size based on week's max size
                crop_x = max(0, x + width // 2 - max_width // 2)
                crop_y = max(0, y + height // 2 - max_height // 2)
                crop_x = min(crop_x, image.shape[1] - max_width)
                crop_y = min(crop_y, image.shape[0] - max_height)
                
                # Extract the cropped region
                cropped_img = image[crop_y:crop_y + max_height, crop_x:crop_x + max_width]
                cropped_mask = mask[crop_y:crop_y + max_height, crop_x:crop_x + max_width]
                
                # Save with numbered format
                base_name = img_name.rsplit('.', 1)[0]
                cropped_img_name = f"{base_name}_{obj_count:02}.jpg"
                cropped_mask_name = f"{base_name}_{obj_count:02}.png"
                
                imwrite(join(output_root, cropped_dir_name, cropped_img_name), cropped_img)
                imwrite(join(output_root, mask_dir_name, cropped_mask_name), cropped_mask)
                
                print(f"Processed: {cropped_img_name}")
                obj_count += 1

if __name__ == "__main__":
    process_images()
