import cv2
import numpy as np
import os
import shutil

# Root directory
root_dir = "./data-test"
image_dir = os.path.join(root_dir, "image")
mask_dir = os.path.join(root_dir, "mask")
output_dir = os.path.join(root_dir, "output")

# Effects to apply
effects = ["grayscale", "hue_up", "hue_down", "contrast_up", "contrast_down", "dying"]

# Clear output folder before each run
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Remove everything
os.makedirs(output_dir, exist_ok=True)

def apply_effects(image, mask):
    """Applies grayscale, hue shift (up & down), contrast change (up & down), and dying plant effect."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- Grayscale Effect ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image_gray = np.where(mask[:, :, None] == 255, gray_colored, image)

    # --- Hue Shift Up ---
    h_up = (h + 15) % 180
    hsv_up = cv2.merge([h_up, s, v])
    image_hue_up = np.where(mask[:, :, None] == 255, cv2.cvtColor(hsv_up, cv2.COLOR_HSV2BGR), image)

    # --- Hue Shift Down ---
    h_down = (h - 15) % 180
    hsv_down = cv2.merge([h_down, s, v])
    image_hue_down = np.where(mask[:, :, None] == 255, cv2.cvtColor(hsv_down, cv2.COLOR_HSV2BGR), image)

    # --- Contrast Increase ---
    alpha_up = 1.5  # Increase contrast
    contrast_up = cv2.convertScaleAbs(image, alpha=alpha_up, beta=0)
    image_contrast_up = np.where(mask[:, :, None] == 255, contrast_up, image)

    # --- Contrast Decrease ---
    alpha_down = 0.5  # Decrease contrast
    contrast_down = cv2.convertScaleAbs(image, alpha=alpha_down, beta=0)
    image_contrast_down = np.where(mask[:, :, None] == 255, contrast_down, image)

    # --- Dying Plant Effect (Green to Brown) ---
    brown_h = np.clip(h - 20, 10, 30)
    brown_s = np.clip(s - 40, 50, 255)
    brown_v = np.clip(v - 20, 50, 255)

    hsv_brown = cv2.merge([brown_h, brown_s, brown_v])
    image_brown = np.where(mask[:, :, None] == 255, cv2.cvtColor(hsv_brown, cv2.COLOR_HSV2BGR), image)

    return image_gray, image_hue_up, image_hue_down, image_contrast_up, image_contrast_down, image_brown

# Process each subfolder (e.g., week3, week8, week12, week18)
for week_folder in sorted(os.listdir(image_dir)):
    week_image_path = os.path.join(image_dir, week_folder)
    week_mask_path = os.path.join(mask_dir, week_folder)

    if not os.path.isdir(week_image_path) or not os.path.isdir(week_mask_path):
        continue  # Skip non-folder files

    # Create output subfolders for this week
    for effect in effects:
        os.makedirs(os.path.join(output_dir, effect, week_folder), exist_ok=True)

    # Process all images in this week's folder
    for filename in os.listdir(week_image_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(week_image_path, filename)
            mask_path = os.path.join(week_mask_path, filename)  # Assume mask has the same filename

            if not os.path.exists(mask_path):
                print(f"Warning: Mask for {filename} in {week_folder} not found, skipping...")
                continue

            # Load image and mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Error loading {filename} in {week_folder}, skipping...")
                continue

            # Ensure binary mask
            _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

            # Apply Effects
            image_gray, image_hue_up, image_hue_down, image_contrast_up, image_contrast_down, image_brown = apply_effects(image, mask)

            # Save outputs
            cv2.imwrite(os.path.join(output_dir, "grayscale", week_folder, filename), image_gray)
            cv2.imwrite(os.path.join(output_dir, "hue_up", week_folder, filename), image_hue_up)
            cv2.imwrite(os.path.join(output_dir, "hue_down", week_folder, filename), image_hue_down)
            cv2.imwrite(os.path.join(output_dir, "contrast_up", week_folder, filename), image_contrast_up)
            cv2.imwrite(os.path.join(output_dir, "contrast_down", week_folder, filename), image_contrast_down)
            cv2.imwrite(os.path.join(output_dir, "dying", week_folder, filename), image_brown)

            print(f"Processed: {week_folder}/{filename}")

print("Processing complete! Outputs saved in './data-test/output/'")
