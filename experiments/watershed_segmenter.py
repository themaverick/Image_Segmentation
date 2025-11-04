import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from collections import deque

# 1. Compute local minima as markers
def get_local_minima(gray):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(gray, kernel)
    minima = (gray == eroded)
    return minima.astype(np.uint8)

# 2. Label each connected component (marker)
def label_markers(minima):
    num_labels, markers = cv2.connectedComponents(minima)
    return markers, num_labels

# 3. Watershed from scratch
def watershed_from_scratch(gray, markers):
    h, w = gray.shape
    # Constants
    WATERSHED = -1
    INIT = -2

    # Initialize label and visited map
    label_map = np.full((h, w), INIT, dtype=np.int32)
    label_map[markers > 0] = markers[markers > 0]

    # Priority queue for pixels: (intensity, y, x)
    pq = []

    # Populate queue with boundary of initial markers
    for y in range(h):
        for x in range(w):
            if markers[y, x] > 0:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if markers[ny, nx] == 0 and label_map[ny, nx] == INIT:
                                heapq.heappush(pq, (gray[ny, nx], ny, nx))
                                label_map[ny, nx] = 0  # Mark as in queue

    # Flooding
    while pq:
        intensity, y, x = heapq.heappop(pq)

        neighbor_labels = set()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    lbl = label_map[ny, nx]
                    if lbl > 0:
                        neighbor_labels.add(lbl)

        if len(neighbor_labels) == 1:
            label_map[y, x] = neighbor_labels.pop()
        elif len(neighbor_labels) > 1:
            label_map[y, x] = WATERSHED

        # Add unvisited neighbors to the queue
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if label_map[ny, nx] == INIT:
                        heapq.heappush(pq, (gray[ny, nx], ny, nx))
                        label_map[ny, nx] = 0  # Mark as in queue

    return label_map

import numpy as np
import cv2
import heapq

def improved_watershed(image_path):
    # Load and preprocess image
    original = cv2.imread(image_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
   
    # Step 1: Better marker detection using adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 4)
   
    # Step 2: Noise removal and sure background area
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
   
    # Step 3: Distance transform for better foreground detection
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
   
    # Step 4: Create markers using connected components
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1  # Add 1 to all labels so background is 1
   
    # Step 5: Apply custom watershed algorithm
    label_map = watershed_from_scratch(blurred, markers)
   
    # Enhanced visualization
    output = original.copy()
    boundaries = (label_map == -1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(boundaries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0,0,255), 1)
   
    # Create intermediate step visualization
    process_steps = {
        "Original": original,
        "Blurred": cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
        "Threshold": cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
        "Foreground Markers": cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR),
        "Final Segmentation": output
    }
   
    return process_steps

def watershed_from_scratch(gray, markers):
    h, w = gray.shape
    WATERSHED = -1
    INIT = -2
   
    label_map = np.full((h, w), INIT, dtype=np.int32)
    label_map[markers > 1] = markers[markers > 1]  # Skip background marker
   
    pq = []
    # Initialize queue with marker boundaries
    for y in range(h):
        for x in range(w):
            if label_map[y, x] > 0:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if label_map[ny, nx] == INIT:
                                heapq.heappush(pq, (gray[ny, nx], ny, nx))
                                label_map[ny, nx] = 0  # Queued

    # Improved flooding with gradient consideration
    while pq:
        intensity, y, x = heapq.heappop(pq)
        neighbors = []
       
        # Check 8 neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbors.append(label_map[ny, nx])
       
        # Find unique labels excluding watershed and background
        unique = set(n for n in neighbors if n > 0)
       
        if len(unique) == 0:
            label_map[y, x] = 1  # Background
        elif len(unique) == 1:
            label_map[y, x] = unique.pop()
        else:
            label_map[y, x] = WATERSHED

        # Add neighbors to queue
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w:
                    if label_map[ny, nx] == INIT:
                        heapq.heappush(pq, (gray[ny, nx], ny, nx))
                        label_map[ny, nx] = 0

    return label_map

# Gradio integration would use:
def generate_watershed(image_path):
    results = improved_watershed(image_path)
    return (
        results["Original"],
        results["Blurred"],
        results["Threshold"],
    )


if __name__ == "__main__":
    # Run the process
    # Load grayscale image
    image = cv2.imread("/home/akshat/projects/CSL7360_Project/bird.jpeg", cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    minima = get_local_minima(image)
    markers, num_labels = label_markers(minima)
    result = watershed_from_scratch(image, markers)

    # Visualization
    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    output[result == -1] = [255, 0, 0]  # Watershed lines in red
    output[result > 0] = [0, 255, 0]    # Segments in green
    output[markers > 0] = [0, 0, 255]   # Original minima in blue


    # Save the original grayscale and the output image
    cv2.imwrite("original_grayscale.png", image)
    cv2.imwrite("watershed_output.png", output)

    print("Images saved as 'original_grayscale.png' and 'watershed_output.png'")