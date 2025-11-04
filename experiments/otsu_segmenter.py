import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io

def otsu_threshold(image):
    hist, bin_edges = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist = hist.astype(float)
    total_pixels = image.size
    pixel_probability = hist / total_pixels

    max_variance = 0
    optimal_threshold = 0

    for threshold in range(1, 256):
        weight_background = np.sum(pixel_probability[:threshold])
        weight_foreground = np.sum(pixel_probability[threshold:])

        if weight_background == 0 or weight_foreground == 0:
            continue

        mean_background = np.sum(np.arange(threshold) * pixel_probability[:threshold]) / weight_background
        mean_foreground = np.sum(np.arange(threshold, 256) * pixel_probability[threshold:]) / weight_foreground

        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance > max_variance:
            max_variance = variance
            optimal_threshold = threshold

    segmented_image = np.zeros_like(image)
    segmented_image[image >= optimal_threshold] = 255

    return optimal_threshold, segmented_image

def generate_segmented_image(image_path):
    # Convert PIL to OpenCV format
    print(f"Image path: {image_path}")
    image = Image.open(image_path)
    image_np = np.array(image)
    original_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if len(original_image.shape) == 3:
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = original_image.copy()

    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Our implementation
    our_threshold, our_segmented = otsu_threshold(blurred)

    # OpenCV's implementation
    opencv_threshold, opencv_segmented = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Create histogram figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(gray_image.ravel(), 256, [0, 256], color='gray')
    ax.axvline(x=our_threshold, color='red', linestyle='--', label=f'Ours: {our_threshold}')
    ax.axvline(x=opencv_threshold, color='green', linestyle='--', label=f'OpenCV: {opencv_threshold}')
    ax.set_title("Histogram with Thresholds")
    ax.legend()

    # Convert Matplotlib figure to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    hist_image = Image.open(buf)
    plt.close(fig)  # Close the figure to free memory


    return (
        image, 
        Image.fromarray(our_segmented), 
        Image.fromarray(opencv_segmented), 
        hist_image,
        f"Our Threshold: {our_threshold}\nOpenCV Threshold: {opencv_threshold}", 
    )
if __name__ == "__main__":
    #example usage
    # Ensure you have the image path set correctly
    image_path = '/home/akshat/projects/CSL7360_Project/bird.jpeg'  
    image = cv2.imread('/home/akshat/projects/CSL7360_Project/bird.jpeg') 
    # Call the function
    generate_segmented_image(image)



# # Optionally, save results to files
# cv2.imwrite("our_segmented.png", our_segmented)
# cv2.imwrite("opencv_segmented.png", opencv_segmented)

