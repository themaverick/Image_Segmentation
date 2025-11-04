import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io

def initialize_centroids(data, K):
    """Randomly choose K data points as initial centroids."""
    indices = np.random.choice(data.shape[0], K, replace=False)
    return data[indices]

def compute_distances(data, centroids):
    """Compute the Euclidean distance between each data point and each centroid."""
    return np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

def update_centroids(data, labels, K):
    """Update centroids as the mean of the points assigned to each cluster."""
    new_centroids = np.zeros((K, data.shape[1]))
    for k in range(K):
        cluster_points = data[labels == k]
        if len(cluster_points) > 0:
            new_centroids[k] = np.mean(cluster_points, axis=0)
    return new_centroids

def kmeans_from_scratch(image, K=4, max_iters=100, tol=1e-4):
    """Apply K-means clustering from scratch to segment the image."""
    data = image.reshape((-1, 3)).astype(np.float32)

    centroids = initialize_centroids(data, K)

    for i in range(max_iters):
        distances = compute_distances(data, centroids)
        labels = np.argmin(distances, axis=1)

        new_centroids = update_centroids(data, labels, K)
        shift = np.linalg.norm(new_centroids - centroids)

        if shift < tol:
            break
        centroids = new_centroids

    segmented_data = centroids[labels].astype(np.uint8)
    segmented_image = segmented_data.reshape(image.shape)

    return segmented_image, labels.reshape(image.shape[:2]), centroids.astype(np.uint8)

def generate_kmeans_segmented_image(image_path, k=3):
    """Process image with K-means for Gradio app"""
    image = Image.open(image_path)
    image_np = np.array(image)
    
    if len(image_np.shape) == 3:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB) 
    else:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    seg_img, labels, centers = kmeans_from_scratch(image_rgb, K=k)
    
    colors_image = np.zeros((50 * k, 100, 3), dtype=np.uint8)
    for i, color in enumerate(centers):
        colors_image[i*50:(i+1)*50, :] = color
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(seg_img)
    axes[1].set_title(f"K-Means (K={k})")
    axes[1].axis('off')
    
    axes[2].imshow(colors_image)
    axes[2].set_title("Cluster Colors")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    comparison_image = Image.open(buf)
    plt.close(fig)
    
    return image, Image.fromarray(seg_img), comparison_image, f"K-Means clustering with K={k}"

if __name__ == "__main__":
    image_path = "/home/akshat/projects/CSL7360_Project/bird.jpeg"
    original, segmented, comparison, text = generate_kmeans_segmented_image(image_path, k=3)
    
    # Save output images instead of displaying them
    segmented.save("kmeans_segmented.png")
    comparison.save("kmeans_comparison.png")
    print(text)