import gradio as gr
import torch
from torchvision import transforms
from experiments.otsu_segmenter import generate_segmented_image
from experiments.kmeans_segmenter import generate_kmeans_segmented_image
from experiments.enhanced_kmeans_segmenter import slic_kmeans
from experiments.watershed_segmenter import generate_watershed
from experiments.felzenszwalb_segmentation import segment
from experiments.SegNet.efficient_b0_backbone.architecture import SegNetEfficientNet, NUM_CLASSES, DEVICE
from experiments.SegNet.vgg_backbone.model import SegNet
# from experiments.ensemble_method import generate_ensemble_segmentation
import numpy as np
from PIL import Image
from matplotlib import cm
import gdown
import os

# Check if the saved_models directory exists, if not create it
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

# Check if the model file already exists before downloading
if not os.path.exists("saved_models/segnet_vgg.pth"):
    print("Downloading SegNet VGG weights...")
    segnet_vgg_weights = "https://drive.google.com/file/d/1EFXKQ_3bDW9FbZCqOLdrE0DOI0V4W82o/view?usp=sharing"
    gdown.download(segnet_vgg_weights, "saved_models/segnet_vgg.pth", fuzzy=True)
    print("Download complete!")
else:
    print("SegNet VGG weights already exist, skipping download.")

def generate_segnet_vgg(image_path):
    model = SegNet(32).to(DEVICE)
    model.load_state_dict(torch.load("saved_models/segnet_vgg.pth", map_location=DEVICE))
    # Set model to evaluation mode
    model.eval()
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    # Apply same preprocessing as during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size to match your model's expected input
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Convert prediction to visualization
    # Option 1: Use a colormap for visualization
    colormap = cm.get_cmap('nipy_spectral')
    colored_mask = colormap(pred_mask / (pred_mask.max() or 1))  # Normalize, handle case where max is 0
    colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)  # Drop alpha and convert to uint8
    segmented_image = Image.fromarray(colored_mask)
    
    # Resize segmented image to match original image size
    segmented_image = segmented_image.resize(original_image.size, Image.NEAREST)
    
    return original_image, segmented_image

def generate_kmeans(image_path,k):
    kmeans_image_output, kmeans_segmented_image_output,_,kmeans_threshold_text=generate_kmeans_segmented_image(image_path, k)
    return kmeans_image_output, kmeans_segmented_image_output, kmeans_threshold_text

def generate_slic(image_path,k,m,max_iter):
    image,seg_img, labels, centers = slic_kmeans(image_path, K=k, m=m, max_iter=max_iter)
    return image,seg_img

def generate_felzenszwalb(image_path, sigma, k, min_size_factor):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    segments_fz = segment(image_np, sigma=sigma, k=k, min_size=min_size_factor)
    segments_fz = segments_fz.astype(np.uint8)
    
    return image, segments_fz

def SegNet_efficient_b0(image_path):
    model = SegNetEfficientNet(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("saved_models/segnet_efficientnet_camvid.pth", map_location=DEVICE))
    model.eval()
    transform = transforms.Compose([
    transforms.Resize((360, 480)),  # Or larger if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Convert original image for Gradio display
    original_image_resized = image

    # Convert predicted mask to a color image using a colormap
    colormap = cm.get_cmap('nipy_spectral')
    colored_mask = colormap(pred_mask / pred_mask.max())  # Normalize
    colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)  # Drop alpha and convert to uint8
    mask_pil = Image.fromarray(colored_mask)

    return original_image_resized, mask_pil

def ensemble_segmentation(image_path):
    """
    Ensemble segmentation combining SegNet and Otsu,
    assuming Otsu produces a mask with the foreground as black (value 0)
    and background as white (value 255).

    In this ensemble, we force the SegNet prediction to background (class 0)
    where Otsu indicates background (after inversion, i.e., where otsu_bin==0).

    Parameters:
        image_path (str): Path to the input image.
        
    Returns:
        original_image: The original resized image used for segmentation.
        segnet_mask_pil: SegNet multi-class segmentation output (PIL image).
        otsu_mask_pil: The original Otsu binary segmentation mask (PIL image).
        ensemble_mask_pil: Final ensemble segmentation mask (PIL image).
    """
    # Run SegNet segmentation (model outputs a multi-class mask).
    segnet_orig, segnet_mask_pil = SegNet_efficient_b0(image_path)
    # Convert SegNet output to a NumPy array (assumed grayscale labeling, e.g., background=0).
    segnet_mask_np = np.array(segnet_mask_pil.convert("L"))

    # Run Otsu segmentation. (generate_segmented_image returns several outputs.)
    _, otsu_segmented_pil, _, _, _ = generate_segmented_image(image_path)
    
    # Resize Otsu mask to match SegNet output shape, e.g., (480, 360) if SegNet works in that resolution.
    resized_shape = (segnet_mask_np.shape[1], segnet_mask_np.shape[0])
    otsu_mask_resized = otsu_segmented_pil.resize(resized_shape, Image.NEAREST)
    otsu_mask_np = np.array(otsu_mask_resized)
    
    # Invert Otsu's binary mask:
    # Assuming that in otsu_mask_np, foreground is black (0) and background is white (255),
    # we build a binary mask where "1" represents the object's area.
    otsu_bin = (otsu_mask_np == 0).astype(np.uint8)  # Now, foreground is 1 and background is 0.
    
    # Create the ensemble segmentation:
    # Where Otsu indicates foreground (otsu_bin==1), keep SegNet's prediction;
    # where Otsu is background (otsu_bin==0), force it to background class (0).
    ensemble_seg = np.where(otsu_bin == 1, segnet_mask_np, 0)
    
    # Convert back to a PIL image.
    ensemble_mask_pil = Image.fromarray(ensemble_seg.astype(np.uint8))
    
    return segnet_orig, segnet_mask_pil, otsu_segmented_pil, ensemble_mask_pil

with gr.Blocks() as demo:
    gr.Markdown("# Image Segmentation using Classical CV")
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Otsu's Method"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Image File")
                    display_btn = gr.Button("Segment this image")
                    threshold_text = gr.Textbox(label="Threshold Comparison", value="", interactive=False)
                
                with gr.Column(scale=2):
                    image_output = gr.Image(label="Original Image")
                    histogram_output = gr.Image(label="Histogram")
                    segmented_image_output = gr.Image(label="Our Segmented Image")
                    opencv_segmented_image_output = gr.Image(label="OpenCV Segmented Image")
            display_btn.click(
                fn=generate_segmented_image,
                inputs=file_input,
                outputs=[image_output, segmented_image_output, opencv_segmented_image_output, histogram_output, threshold_text]
            )
        with gr.TabItem("K-means Segmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    kmeans_file_input = gr.File(label="Upload Image File")
                    kmeans_k_value = gr.Slider(minimum=2, maximum=10, value=3, step=1, label="Number of Clusters (K)")
                    kmeans_display_btn = gr.Button("Segment this image")
                    kmeans_threshold_text = gr.Textbox(label="K-means Info", value="", interactive=False)
                
                with gr.Column(scale=2):
                    kmeans_image_output = gr.Image(label="Original Image")
                    kmeans_segmented_image_output = gr.Image(label="K-means Segmented Image")
            
            kmeans_display_btn.click(
                fn=generate_kmeans,
                inputs=[kmeans_file_input, kmeans_k_value],
                outputs=[kmeans_image_output, kmeans_segmented_image_output, kmeans_threshold_text]
        )
        with gr.TabItem("SLIC Segmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    slic_file_input = gr.File(label="Upload Image File")
                    slic_k_value = gr.Slider(minimum=2, maximum=200, value=3, step=1, label="Number of superpixels")
                    slic_m_value = gr.Slider(minimum=1, maximum=40, value=3, step=1, label="Compactness factor")
                    slic_max_iter_value = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of iterations")
                    slic_display_btn = gr.Button("Segment this image")
                
                with gr.Column(scale=2):
                    slic_image_output = gr.Image(label="Original Image",container=True)
                    slic_segmented_image_output = gr.Image(label="SLIC Segmented Image",container=True)
            
            slic_display_btn.click(
                fn=generate_slic,
                inputs=[slic_file_input, slic_k_value,slic_m_value,slic_max_iter_value],
                outputs=[slic_image_output,slic_segmented_image_output]
        )
            
        with gr.TabItem("Watershed"):
            with gr.Row():
                with gr.Column(scale=1):
                    watershed_file = gr.File(label="Upload Image")
                    watershed_btn = gr.Button("Run Watershed")
               
                with gr.Column(scale=2):
                    original_img = gr.Image(label="1. Original")
                    blurred_img = gr.Image(label="2. Blurred")
                    threshold_img = gr.Image(label="3. Threshold")

            watershed_btn.click(
                fn=generate_watershed,
                inputs=[watershed_file],
                outputs=[original_img, blurred_img, threshold_img]
            )
        with gr.TabItem("Felzenszwalb Algorithm Segmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    felzenszwalb_file_input = gr.File(label="Upload Image File")
                    sigma_value = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.1, label="Sigma")
                    K_value = gr.Slider(minimum=2, maximum=1000, value=2, step=1, label="K value")
                    min_size_value = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Min Size Factor")
                    felzenszwalb_display_btn = gr.Button("Segment this image")
                
                with gr.Column(scale=2):
                    felzenszwalb_image_output = gr.Image(label="Original Image",container=True)
                    felzenszwalb_segmented_image_output = gr.Image(label="felzenszwalb Segmented Image",container=True)
            
            felzenszwalb_display_btn.click(
                fn=generate_felzenszwalb,
                inputs=[felzenszwalb_file_input,sigma_value,K_value,min_size_value],
                outputs=[felzenszwalb_image_output,felzenszwalb_segmented_image_output]
        )
        with gr.TabItem("SegNet EfficientNet B0 Segmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    segnet_file_input = gr.File(label="Upload Image File")
                    segnet_display_btn = gr.Button("Segment this image")
                
                with gr.Column(scale=2):
                    segnet_image_output = gr.Image(label="Original Image")
                    segnet_segmented_image_output = gr.Image(label="SegNet Segmented Image")
            
            segnet_display_btn.click(
                fn=SegNet_efficient_b0,
                inputs=[segnet_file_input],
                outputs=[segnet_image_output,segnet_segmented_image_output]
        )
        with gr.TabItem("SegNet VGG Segmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    segnet_file_input = gr.File(label="Upload Image File")
                    segnet_display_btn = gr.Button("Segment this image")
                
                with gr.Column(scale=2):
                    segnet_image_output = gr.Image(label="Original Image")
                    segnet_segmented_image_output = gr.Image(label="SegNet VGG Segmented Image")
            
            segnet_display_btn.click(
                fn=generate_segnet_vgg,
                inputs=[segnet_file_input],
                outputs=[segnet_image_output,segnet_segmented_image_output]
        )
        # In app.py
        with gr.TabItem("Ensemble Segmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    ensemble_file_input = gr.File(label="Upload Image File")
                    ensemble_display_btn = gr.Button("Segment with Ensemble Method")
                
                with gr.Column(scale=2):
                    ensemble_image_output = gr.Image(label="Original Image")
                    ensemble_mask = gr.Image(label="Ensemble Segmented Image")
                    ensemble_segnet_segmented_output = gr.Image(label="SegNet Efficient B0 Segmented Image")
                    ensemble_otsu_segmented_output = gr.Image(label="Otsu Segmented Image")
            
            ensemble_display_btn.click(
                fn=ensemble_segmentation,
                inputs=[ensemble_file_input],
                outputs=[ensemble_image_output, ensemble_segnet_segmented_output, ensemble_otsu_segmented_output, ensemble_mask]
            )

if __name__ == "__main__":
    demo.launch()

