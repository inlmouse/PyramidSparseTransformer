import cv2
from ultralytics import YOLO
from ultralytics.solutions import Heatmap
import numpy as np
import os
import random
import time

# Function to load models from specified paths
def load_models(model_paths):
    """
    Load YOLO models and their corresponding heatmap instances.

    Args:
        model_paths (list): List of paths to model files.

    Returns:
        tuple: List of loaded models and corresponding Heatmap objects.
    """
    models = []
    heatmaps = []
    for path in model_paths:
        try:
            model = YOLO(path)
            heatmap = Heatmap(model=path, colormap=cv2.COLORMAP_JET)
            models.append(model)
            heatmaps.append(heatmap)
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")
    return models, heatmaps

# Function to randomly select images from a folder
def get_random_images(folder_path, num_images):
    """
    Randomly select a specified number of image files from a folder.

    Args:
        folder_path (str): Path to the folder containing images.
        num_images (int): Number of images to select.

    Returns:
        list: List of full paths to selected image files.

    Raises:
        ValueError: If the number of available images is less than requested.
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) < num_images:
        raise ValueError(f"Only {len(image_files)} images found in folder, less than requested {num_images}.")
    return [os.path.join(folder_path, f) for f in random.sample(image_files, num_images)]

# Function to resize an image to 320x320
def resize_image(image):
    """
    Resize an image to a fixed resolution of 320x320.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Resized image.
    """
    return cv2.resize(image, (320, 320))

# Function to generate a dense heatmap based on confidence grid
def generate_dense_heatmap(image, results):
    """
    Generate a dense heatmap based on confidence scores from detection results.

    Args:
        image (np.ndarray): Input image.
        results: Detection results from YOLO model.

    Returns:
        np.ndarray: Overlay of heatmap on the original image.
    """
    h, w = image.shape[:2]

    # Extract confidence grid from prediction results
    result = results[0]
    if hasattr(result, 'pred') and result.pred is not None:
        pred = result.pred  # Prediction grid, shape typically [batch, anchors, grid_h, grid_w, num_outputs]
        conf_grid = pred[0, :, :, :, 4].cpu().numpy()  # Extract confidence scores (assuming 4th channel)
        conf_grid = np.max(conf_grid, axis=0)  # Aggregate by taking maximum
    else:
        conf_grid = np.zeros((h // 16, w // 16), dtype=np.float32)  # Default grid size 1/16 of image
        for box in result.boxes:
            conf = box.conf.item()
            cx, cy = map(int, (box.xywh[0, 0], box.xywh[0, 1]))  # Center coordinates
            grid_x, grid_y = cx // 16, cy // 16  # Map to grid
            if 0 <= grid_y < conf_grid.shape[0] and 0 <= grid_x < conf_grid.shape[1]:
                conf_grid[grid_y, grid_x] = max(conf_grid[grid_y, grid_x], conf)

    # Resize heatmap to match original image dimensions
    heatmap = cv2.resize(conf_grid, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize heatmap to [0, 255]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Apply color mapping
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)
    return overlay

# Function to draw detection results on an image
def draw_detections(image, results):
    """
    Draw bounding boxes and labels on an image based on detection results.

    Args:
        image (np.ndarray): Input image.
        results: Detection results from YOLO model.

    Returns:
        np.ndarray: Image with drawn detections.
    """
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{result.names[int(box.cls)]} {float(box.conf):.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Main function to process images and generate visualizations
def main():
    """
    Main function to load models, process random images, and generate detection visualizations.
    """
    # Specify image folder and model paths (replace with your actual paths)
    image_folder = "./datasets/images/val2017"  # Placeholder for image folder
    model_paths = ["./models/yolov8n.pt", "./models/yolo11n.pt", "./models/best.pt"]  # Placeholder for model files

    # Load three models
    models, heatmaps = load_models(model_paths)
    if len(models) != 3:
        print("Failed to load all models, exiting program.")
        return

    # Create save folder
    save_folder = "./visualization/Grad-CAM-val2017"
    os.makedirs(save_folder, exist_ok=True)

    # Randomly select 100 images
    image_paths = get_random_images(image_folder, 100)

    # Process each image
    for idx, image_path in enumerate(image_paths):
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Failed to read image: {image_path}")
            continue

        # Resize original image
        resized_original = resize_image(original_image)

        # Perform detection with three models
        heatmap_images = []  # Store Grad-CAM heatmaps
        for model, heatmap in zip(models, heatmaps):
            results = model.predict(original_image)
            frame_with_heat = heatmap(original_image)
            heatmap_images.append(resize_image(frame_with_heat.plot_im))

        # Concatenate original image with heatmap results horizontally
        concat_image = cv2.hconcat([resized_original] + heatmap_images)

        # Add titles to the concatenated image
        titles = ["Original", "YOLOv8n", "YOLOv11n", "YOLOv11n-PST"]
        for i, title in enumerate(titles):
            cv2.putText(concat_image, title, (i * 320 + 10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save the concatenated image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_folder, f"detection_{idx+1}_{timestamp}.jpg")
        cv2.imwrite(save_path, concat_image)
        print(f"Saved image {idx+1}/100 to {save_path}")

    print("All images processed successfully!")

if __name__ == "__main__":
    main()