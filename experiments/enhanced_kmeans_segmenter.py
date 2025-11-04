import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

def slic_kmeans(image_path, K=100, m=10, max_iter=10):
    """
    Perform superpixel segmentation using enhanced K-means with LAB+XY.
    Args:
        image (np.ndarray): RGB input image.
        K (int): Number of superpixels.
        m (float): Compactness factor.
        max_iter (int): Number of iterations.
    Returns:
        segmented_img: The segmented image with cluster colors.
        labels: Cluster label for each pixel.
    """
    jpg_image = Image.open(image_path)
    image = np.array(jpg_image)
    h, w = image.shape[:2]
    S = int(np.sqrt(h * w / K))  # grid interval

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Create 5D feature vector [L, a, b, x, y]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    features = np.dstack((lab, X, Y)).reshape((-1, 5))

    # Initialize cluster centers on grid
    centers = []
    for y in range(S // 2, h, S):
        for x in range(S // 2, w, S):
            center = features[y * w + x]
            centers.append(center)
    centers = np.array(centers)

    labels = np.full((h * w,), -1, dtype=np.int32)
    distances = np.full((h * w,), np.inf)

    for iteration in tqdm(range(max_iter)):
        for idx, center in enumerate(centers):
            l, a, b, cx, cy = center
            x_start, x_end = max(0, int(cx - S)), min(w, int(cx + S))
            y_start, y_end = max(0, int(cy - S)), min(h, int(cy + S))

            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    i = y * w + x
                    fp = features[i]
                    dc = np.linalg.norm(fp[:3] - center[:3])  # LAB distance
                    ds = np.linalg.norm(fp[3:] - center[3:])  # XY distance
                    D = np.sqrt(dc**2 + (ds / S)**2 * m**2)

                    if D < distances[i]:
                        distances[i] = D
                        labels[i] = idx

        # Update cluster centers
        new_centers = np.zeros_like(centers)
        count = np.zeros(len(centers))
        for i in range(h * w):
            lbl = labels[i]
            new_centers[lbl] += features[i]
            count[lbl] += 1
        for i in range(len(centers)):
            if count[i] > 0:
                new_centers[i] /= count[i]
        centers = new_centers

    # Recolor image based on cluster centers
    segmented_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h * w):
        lbl = labels[i]
        lab_val = centers[lbl][:3]
        lab_pixel = np.uint8([[lab_val]])
        rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)[0][0]
        segmented_img[i // w, i % w] = rgb_pixel

    return jpg_image, Image.fromarray(segmented_img), labels.reshape((h, w)), centers

# img_path = "/home/akshat/projects/CSL7360_Project/bird.jpeg"
# image = cv2.imread(img_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# _,seg_img, labels, centers = slic_kmeans(image, K=2, m=20)
# seg_img.save("enhaned_kmeans_segmented.png")
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(seg_img)
# plt.title("SLIC-like K-Means Segmentation")
# plt.axis("off")
# plt.tight_layout()
# plt.show()
