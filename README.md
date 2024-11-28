# Document Scanner Application

This script processes an image of a scanned document, identifies its boundaries, and applies a perspective transformation to produce a clean, top-down view of the document. The code is particularly useful for automating document digitization tasks, such as archiving paper forms or enhancing scanned images for Optical Character Recognition (OCR).

### Input Image
![scanned-form](https://github.com/user-attachments/assets/35281a89-52b1-4de0-ada7-faee69627750)

### Edged Image
![edged_image](https://github.com/user-attachments/assets/3629b877-e4a1-4836-8ca2-7be9c2e97607)

### Output Image
![scanned_output](https://github.com/user-attachments/assets/8f931a19-6bba-48ab-a332-246863531d43)

## Features

* Resizes and preprocesses the image.
* Detects document boundaries using edge detection and contour analysis.
* Applies a perspective transformation for a flattened, top-down view of the document.



## Code Walkthrough

### 1. Helper Function: mapp
The mapp function reorders the four corners of a contour to ensure consistent perspective transformation. It identifies:

`Top-left`
`Top-right`
`Bottom-right`
`Bottom-left`
Implementation:
```python
def mapp(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)  # Sum of x and y coordinates
    hnew[0] = h[np.argmin(add)]  # Top-left corner
    hnew[2] = h[np.argmax(add)]  # Bottom-right corner

    diff = np.diff(h, axis=1)  # Difference between x and y
    hnew[1] = h[np.argmin(diff)]  # Top-right corner
    hnew[3] = h[np.argmax(diff)]  # Bottom-left corner

    return hnew
```

The function reshapes and reorders the points based on their geometric properties.

### 2. Image Preprocessing
The preprocessing steps prepare the image for contour detection.

Resize the Image:
```python
image = cv2.imread("scanned-form.jpg")
image = cv2.resize(image, (1300, 800))
cv2.imwrite("resized_image.jpg", image)
```
Ensures a consistent input size for predictable processing and Resized image saved as resized_image.jpg.

Convert to Grayscale:

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_image.jpg", gray)
```

Simplifies the image for edge detection.

Apply Gaussian Blur:

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite("blurred_image.jpg", blurred)
```

Reduces noise and smooths the image.

Apply Edge Detection:

```python
edged = cv2.Canny(blurred, 30, 50)
cv2.imwrite("edged_image.jpg", edged)
```
Highlights the edges of the document.

### 3. Edge Detection and Contour Analysis
Find Contours:
```python
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
```
Identifies the boundaries of objects in the image.

Identify the Document Contour:
```python
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break
```

The document contour is the largest contour with four corners.

### 4. Perspective Transformation
```python
approx = mapp(target)
pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
op = cv2.getPerspectiveTransform(approx, pts)
dst = cv2.warpPerspective(orig, op, (800, 800))

cv2.imwrite("scanned_output.jpg", dst)
```

Converts the document to a top-down view and Transformed document saved as scanned_output.jpg.




