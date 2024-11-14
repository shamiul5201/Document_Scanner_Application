import cv2
import numpy as np

def mapp(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

# Load and resize image
image = cv2.imread("scanned-form.jpg")
image = cv2.resize(image, (1300, 800))
cv2.imwrite("resized_image.jpg", image)  # Save resized image

# Make a copy of the original image
orig = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_image.jpg", gray)  # Save grayscale image

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite("blurred_image.jpg", blurred)  # Save blurred image

# Apply Canny Edge Detection
edged = cv2.Canny(blurred, 30, 50)
cv2.imwrite("edged_image.jpg", edged)  # Save edge-detected image

# Find contours and sort them
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Detect the document contour
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break

# Apply perspective transformation
approx = mapp(target)
pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
op = cv2.getPerspectiveTransform(approx, pts)
dst = cv2.warpPerspective(orig, op, (800, 800))

cv2.imwrite("scanned_output.jpg", dst)  # Save final scanned output

# Display final output (optional)
cv2.imshow("Scanned", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()