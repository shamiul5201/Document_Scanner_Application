# Document Scanner Application

A simple document scanner built with Python and OpenCV that turns photos of documents into clear, high-quality scans. This script detects the document edges, transforms the perspective, and saves the final result as a neatly scanned document image.

### Input Image
![scanned-form](https://github.com/user-attachments/assets/35281a89-52b1-4de0-ada7-faee69627750)

### Edged Image
![edged_image](https://github.com/user-attachments/assets/3629b877-e4a1-4836-8ca2-7be9c2e97607)

### Output Image
![scanned_output](https://github.com/user-attachments/assets/8f931a19-6bba-48ab-a332-246863531d43)

## Features

- **Automatic Edge Detection**: Detects document boundaries to crop and adjust the perspective.
- **Perspective Transform**: Maps the document to a top-down view.
- **Grayscale and Blur Processing**: Improves edge detection by smoothing out noise.

## How It Works

The script uses a combination of grayscale conversion, Gaussian blur, and Canny edge detection to find the document's contours. It then applies a perspective transform to make the document appear as though it was scanned directly.

## Code Overview

### 1. Import Libraries
```python
import cv2
import numpy as np
```
### 2. Define the helper function
```python
def mapp(h):
    # Rearranges corner points to standard order
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    return hnew
```
The `mapp` function is designed to reorder four corner points of a document so that they’re in a consistent, standard order. When a document's corners are detected, the points can be arranged in any sequence, which can make it difficult to apply transformations. This function rearranges them to a fixed order: top-left, top-right, bottom-right, and bottom-left.

It works by:
1. Calculating the sum of each point’s x and y coordinates to find the top-left (smallest sum) and bottom-right (largest sum).
2. Calculating the difference between each point’s x and y coordinates to identify the top-right (smallest difference) and bottom-left (largest difference).

### 3. Process Image
- **Load and Resize the Image**
- **Convert to Grayscale**
- **Apply Gaussian Blur for Smoother Edges**
- **Use Canny Edge Detection**
