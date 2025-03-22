import cv2
import numpy as np
import os
import shutil

# Root directory
root_dir = "./data-test"
image_dir = os.path.join(root_dir, "image")
mask_dir = os.path.join(root_dir, "mask")
output_dir = os.path.join(root_dir, "output")

# Define effect variations
contrast_levels_up = [1.1, 1.3, 1.5]
contrast_levels_down = [0.9, 0.7, 0.5]
hue_shifts = [5, 15, 30]
dying_variations = [(10, 50, 50), (20, 70, 70), (30, 90, 90)]  # (hue, saturation, value)

# Clear output folder before each run
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Remove everything
os.makedirs(output_dir, exist_ok=True)

def apply_effects(image, mask):
    """Applies grayscale, multiple hue shifts, contrast changes, and multiple dying plant effects."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- Grayscale Effect ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image_gray = np.where(mask[:, :, None] == 255, gray_colored, image)

    # --- Hue Shifts ---
    hue_images = {}
    for shift in hue_shifts:
        h_shifted = (h + shift) % 180
        hsv_shifted = cv2.merge([h_shifted, s, v])
        hue_images[f"hue_up_{shift}"] = np.where(mask[:, :, None] == 255, cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR), image)
    
    for shift in hue_shifts:
        h_shifted = (h - shift) % 180
        hsv_shifted = cv2.merge([h_shifted, s, v])
        hue_images[f"hue_down_{shift}"] = np.where(mask[:, :, None] == 255, cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR), image)

    # --- Contrast Adjustments ---
    contrast_images = {}
    for alpha in contrast_levels_up:
        contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        contrast_images[f"contrast_up_{alpha}"] = np.where(mask[:, :, None] == 255, contrast, image)
    
    for alpha in contrast_levels_down:
        contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        contrast_images[f"contrast_down_{alpha}"] = np.where(mask[:, :, None] == 255, contrast, image)

    # --- Dying Plant Effects ---
    dying_images = {}
    for i, (hue_adj, sat_adj, val_adj) in enumerate(dying_variations, start=1):
        brown_h = np.clip(h - hue_adj, 10, 30)
        brown_s = np.clip(s - sat_adj, 50, 255)
        brown_v = np.clip(v - val_adj, 50, 255)
        hsv_brown = cv2.merge([brown_h, brown_s, brown_v])
        dying_images[f"dying_{i}"] = np.where(mask[:, :, None] == 255, cv2.cvtColor(hsv_brown, cv2.COLOR_HSV2BGR), image)

    return image_gray, hue_images, contrast_images, dying_images

# Process each subfolder (e.g., week3, week8, week12, week18)
for week_folder in sorted(os.listdir(image_dir)):
    week_image_path = os.path.join(image_dir, week_folder)
    week_mask_path = os.path.join(mask_dir, week_folder)

    if not os.path.isdir(week_image_path) or not os.path.isdir(week_mask_path):
        continue  # Skip non-folder files

    # Create output subfolders for this week
    os.makedirs(os.path.join(output_dir, "grayscale", week_folder), exist_ok=True)
    for effect in hue_shifts:
        os.makedirs(os.path.join(output_dir, f"hue_up_{effect}", week_folder), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f"hue_down_{effect}", week_folder), exist_ok=True)
    for effect in contrast_levels_up:
        os.makedirs(os.path.join(output_dir, f"contrast_up_{effect}", week_folder), exist_ok=True)
    for effect in contrast_levels_down:
        os.makedirs(os.path.join(output_dir, f"contrast_down_{effect}", week_folder), exist_ok=True)
    for i in range(1, len(dying_variations) + 1):
        os.makedirs(os.path.join(output_dir, f"dying_{i}", week_folder), exist_ok=True)

    # Process all images in this week's folder
    image_filenames = {os.path.splitext(f)[0]: f for f in os.listdir(week_image_path) if f.endswith((".jpg", ".png", ".jpeg"))}
    mask_filenames = {os.path.splitext(f)[0]: f for f in os.listdir(week_mask_path) if f.endswith((".jpg", ".png", ".jpeg"))}

    for name, image_file in image_filenames.items():
        if name not in mask_filenames:
            print(f"Warning: Mask for {image_file} in {week_folder} not found, skipping...")
            continue

        image_path = os.path.join(week_image_path, image_file)
        mask_path = os.path.join(week_mask_path, mask_filenames[name])
        image_ext = os.path.splitext(image_file)[1]  # Get original image extension

        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error loading {image_file} in {week_folder}, skipping...")
            continue

        # Ensure binary mask
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        # Apply Effects
        image_gray, hue_images, contrast_images, dying_images = apply_effects(image, mask)

        # Save outputs
        cv2.imwrite(os.path.join(output_dir, "grayscale", week_folder, name + image_ext), image_gray)
        for effect, img in hue_images.items():
            cv2.imwrite(os.path.join(output_dir, effect, week_folder, name + image_ext), img)
        for effect, img in contrast_images.items():
            cv2.imwrite(os.path.join(output_dir, effect, week_folder, name + image_ext), img)
        for effect, img in dying_images.items():
            cv2.imwrite(os.path.join(output_dir, effect, week_folder, name + image_ext), img)

        print(f"Processed: {week_folder}/{image_file}")

print("Processing complete! Outputs saved in './data-test/output/'")
