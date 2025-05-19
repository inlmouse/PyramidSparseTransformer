import cv2
import numpy as np
from ultralytics import YOLO
import os
import random
import time
from PIL import Image
from torchvision import transforms

# Function to load models from specified paths
def load_models(model_paths):
    """
    Load YOLO models from the specified paths.

    Args:
        model_paths (list): List of paths to model files.

    Returns:
        list: List of loaded YOLO models.
    """
    models = []
    for path in model_paths:
        try:
            model = YOLO(path)
            models.append(model)
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")
    return models

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

# Hook function to capture activations from a specific layer
def hook_fn(module, input, output):
    """
    Hook function to capture activations during forward pass.

    Args:
        module: The layer/module being hooked.
        input: Input to the layer.
        output: Output from the layer (activations).

    Modifies:
        activations (dict): Global dictionary to store captured activations.
    """
    activations['feat'] = output.detach()

# Function to generate a heatmap using Grad-CAM
def generate_heatmap(image_path, model):
    """
    Generate a Grad-CAM heatmap for a given image and model.

    Args:
        image_path (str): Path to the input image.
        model: YOLO model for inference.

    Returns:
        np.ndarray: Image with overlaid heatmap.
    """
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    x = tf(img).unsqueeze(0)  # [1, 3, 640, 640]

    # Register hook to capture activations from a target layer
    global activations
    activations = {}
    target_layer = model.model.model[2]  # Target layer in backbone (adjust as needed)
    target_layer.register_forward_hook(hook_fn)

    # Forward pass to capture activations
    _ = model.model(x)  # Run feature extraction
    if 'feat' not in activations:
        print("Failed to capture feature map; target layer may be misconfigured.")
        return np.zeros((640, 640, 3), dtype=np.uint8)

    # Compute heatmap from captured activations
    feat = activations['feat'][0]  # [C, H, W]
    heat = feat.mean(dim=0).cpu().numpy()  # Average across channels to get [H, W]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)  # Normalize to [0, 1]
    heat = (heat * 255).astype(np.uint8)  # Scale to [0, 255]
    heatmap_colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # Apply color mapping

    # Overlay heatmap on the original image
    original_image = cv2.imread(image_path)
    heatmap_resized = cv2.resize(heatmap_colored, (original_image.shape[1], original_image.shape[0]))
    overlay = cv2.addWeighted(original_image, 0.2, heatmap_resized, 0.8, 0)
    return overlay

# Main function to process images and generate heatmaps
def main():
    """
    Main function to load models, process random images, and generate Grad-CAM heatmaps.
    """
    # Specify image folder and model paths (replace with your actual paths)
    image_folder = "./datasets/images/val2017"  # Placeholder for image folder
    model_paths = ["./models/yolov8n.pt", "./models/yolo11n.pt", "./models/best.pt"]  # Placeholder for model files

    # Load three models
    models = load_models(model_paths)
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

        # Generate heatmaps using three models
        heatmap_images = []
        for model in models:
            heatmap_image = generate_heatmap(image_path, model)
            heatmap_images.append(resize_image(heatmap_image))

        # Concatenate original image with heatmaps horizontally
        concat_image = cv2.hconcat([resized_original] + heatmap_images)

        # Add titles to the concatenated image
        titles = ["Original", "YOLOv8n Heatmap", "YOLOv11n Heatmap", "YOLOv11n-PST Heatmap"]
        for i, title in enumerate(titles):
            cv2.putText(concat_image, title, (i * 320 + 10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save the concatenated image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_folder, f"heatmap_{idx+1}_{timestamp}.jpg")
        cv2.imwrite(save_path, concat_image)
        print(f"Saved heatmap {idx+1}/100 to {save_path}")

    print("All heatmaps processed successfully!")

if __name__ == "__main__":
    main()