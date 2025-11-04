import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from experiments.otsu_segmenter import otsu_threshold
from experiments.SegNet.efficient_b0_backbone.architecture import SegNetEfficientNet, NUM_CLASSES, DEVICE

def ensemble_segmentation(image_path, model_path="segnet_efficientnet_voc.pth", boundary_weight=0.3):
    """
    Ensemble segmentation combining Otsu thresholding and SegNet
    
    Args:
        image_path: Path to input image
        model_path: Path to SegNet model weights
        boundary_weight: Weight for boundary refinement (0-1)
        
    Returns:
        original_image: Original input image (PIL)
        ensemble_result: Ensemble segmentation result (PIL)
        method_comparison: Visualization of all methods side by side (PIL)
    """
    # 1. Load the image
    image = Image.open(image_path).convert('RGB')
    original = image.copy()
    image_np = np.array(image)
    
    # 2. Run Otsu thresholding for boundary detection
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold_value, otsu_mask = otsu_threshold(blurred)
    
    # 3. Run SegNet for semantic segmentation
    model = SegNetEfficientNet(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    transform = transforms.Compose([
    transforms.Resize((360, 480)),  # Or larger if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        segnet_pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # 4. Create edge map from Otsu result
    edges = cv2.Canny(otsu_mask, 50, 150)
    
    # Resize to match SegNet output size
    edges_resized = cv2.resize(edges, (segnet_pred.shape[1], segnet_pred.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    # 5. Ensemble: Use Otsu edges to refine SegNet boundaries
    # Create a distance transform from the edges
    dist_transform = cv2.distanceTransform(255 - edges_resized, cv2.DIST_L2, 5)
    dist_transform = dist_transform / dist_transform.max()  # Normalize to 0-1
    
    # Areas close to edges get more influence from Otsu
    edge_weight_map = np.exp(-dist_transform * 5) * boundary_weight
    
    # Create binary mask from SegNet (foreground = any class other than background)
    segnet_binary = (segnet_pred > 0).astype(np.uint8) * 255
    
    # Resize Otsu mask to match SegNet output
    otsu_resized = cv2.resize(otsu_mask, (segnet_pred.shape[1], segnet_pred.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Combine: Use SegNet classes but refine boundaries with Otsu
    # For boundary regions, adjust the segmentation based on Otsu
    refined_binary = segnet_binary.copy()
    boundary_region = edge_weight_map > 0.1
    refined_binary[boundary_region] = (
        (1 - edge_weight_map[boundary_region]) * segnet_binary[boundary_region] + 
        edge_weight_map[boundary_region] * otsu_resized[boundary_region]
    ).astype(np.uint8)
    
    # Apply the refined binary mask to the original SegNet prediction
    ensemble_result = segnet_pred.copy()
    # Where the refined binary is 0, set to background class (0)
    ensemble_result[refined_binary < 128] = 0
    
    # 6. Visualize results
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import io
    
    # Convert semantic maps to color visualizations
    colormap = cm.get_cmap('nipy_spectral')
    
    segnet_colored = colormap(segnet_pred / (NUM_CLASSES - 1))
    segnet_colored = (segnet_colored[:, :, :3] * 255).astype(np.uint8)
    
    ensemble_colored = colormap(ensemble_result / (NUM_CLASSES - 1))
    ensemble_colored = (ensemble_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Resize original image to match the segmentation size
    original_resized = original.resize((segnet_pred.shape[1], segnet_pred.shape[0]))
    
    axes[0].imshow(original_resized)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(otsu_mask, cmap='gray')
    axes[1].set_title(f"Otsu (t={otsu_threshold_value})")
    axes[1].axis('off')
    
    axes[2].imshow(segnet_colored)
    axes[2].set_title("SegNet Prediction")
    axes[2].axis('off')
    
    axes[3].imshow(ensemble_colored)
    axes[3].set_title("Ensemble Result")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Convert the plot to an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    comparison_image = Image.open(buf)
    plt.close(fig)
    
    # Return results
    ensemble_pil = Image.fromarray(ensemble_colored)
    ensemble_pil = ensemble_pil.resize(original.size, Image.NEAREST)
    
    return original, ensemble_pil, comparison_image

# Add this function to your app.py
def generate_ensemble_segmentation(image_path, boundary_weight=0.3):
    """Wrapper for Gradio interface"""
    original, ensemble_result, comparison = ensemble_segmentation(
        image_path, 
        model_path="saved_models/segnet_efficientnet_camvid.pth",
        boundary_weight=boundary_weight
    )
    return original, ensemble_result, comparison