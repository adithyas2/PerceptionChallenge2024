import cv2 as cv
import numpy as np
import scipy.optimize as optimize
from matplotlib import pyplot as plt

# Load image from file
image_path = "/Users/adithyas/Wisconsin Autonomous/red.png"
image = cv.imread(image_path)

# Convert the image to RGB and HSV color spaces
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Define thresholds for detecting red regions in the HSV space
lower_thresh1 = np.array([0, 135, 135])
upper_thresh1 = np.array([15, 255, 255])
lower_thresh2 = np.array([159, 135, 135])
upper_thresh2 = np.array([179, 255, 255])

# Apply the thresholding to capture red pixels
red_mask1 = cv.inRange(image_hsv, lower_thresh1, upper_thresh1)
red_mask2 = cv.inRange(image_hsv, lower_thresh2, upper_thresh2)

# Combine the two masks
red_mask = cv.bitwise_or(red_mask1, red_mask2)

# Apply morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)
cleaned_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
blurred_mask = cv.medianBlur(cleaned_mask, 5)

# Detect edges and contours in the blurred image
edges = cv.Canny(blurred_mask, 70, 255)
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create an empty image for drawing contours
contour_image = np.zeros_like(edges)
cv.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

# Simplify the contours using the Douglas-Peucker algorithm
simplified_contours = []
for contour in contours:
    approx = cv.approxPolyDP(contour, 10, True)
    simplified_contours.append(approx)

# Draw simplified contours
simplified_image = np.zeros_like(edges)
cv.drawContours(simplified_image, simplified_contours, -1, (255, 255, 255), 1)

# Compute convex hulls for each simplified contour
convex_hulls = [cv.convexHull(c) for c in simplified_contours]

# Create an image to display convex hulls
convex_image = np.zeros_like(edges)
cv.drawContours(convex_image, convex_hulls, -1, (255, 255, 255), 2)

# Filter hulls based on the number of points (between 3 and 10)
filtered_hulls = [hull for hull in convex_hulls if 3 <= len(hull) <= 10]

# Create an image for filtered convex hulls
filtered_hulls_image = np.zeros_like(edges)
cv.drawContours(filtered_hulls_image, filtered_hulls, -1, (255, 255, 255), 2)

# Function to check if a convex hull points upward
def is_hull_pointing_up(hull):
    x, y, w, h = cv.boundingRect(hull)
    aspect_ratio = w / h
    if aspect_ratio >= 0.8:
        return False

    vertical_midpoint = y + h / 2
    top_points = [p for p in hull if p[0][1] < vertical_midpoint]
    bottom_points = [p for p in hull if p[0][1] >= vertical_midpoint]

    left_bound = min(p[0][0] for p in bottom_points)
    right_bound = max(p[0][0] for p in bottom_points)

    return all(left_bound <= p[0][0] <= right_bound for p in top_points)

# Identify cones and compute their bounding rectangles
cones, bounding_rects = [], []
for hull in filtered_hulls:
    if is_hull_pointing_up(hull):
        cones.append(hull)
        bounding_rects.append(cv.boundingRect(hull))

# Draw the detected cones and their bounding boxes
cone_image = np.zeros_like(edges)
cv.drawContours(cone_image, cones, -1, (255, 255, 255), 2)

result_image = image_rgb.copy()
cv.drawContours(result_image, cones, -1, (255, 255, 255), 2)

for rect in bounding_rects:
    cv.rectangle(result_image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (1, 255, 1), 3)

# Function to perform least squares linear regression
def perform_least_squares(x_vals, y_vals):
    def linear_func(x, a, b):
        return a * x + b
    params, _ = optimize.curve_fit(linear_func, x_vals, y_vals)
    return params

# Separate the points into left and right based on their x-coordinates
left_side_points = [(r[0] + r[2] / 2, r[1] + r[3] / 2) for r in bounding_rects if r[0] + r[2] / 2 < result_image.shape[1] / 2]
right_side_points = [(r[0] + r[2] / 2, r[1] + r[3] / 2) for r in bounding_rects if r[0] + r[2] / 2 > result_image.shape[1] / 2]

# Fit lines to the left and right points
a_left, b_left = perform_least_squares(np.array([p[0] for p in left_side_points]), np.array([p[1] for p in left_side_points]))
a_right, b_right = perform_least_squares(np.array([p[0] for p in right_side_points]), np.array([p[1] for p in right_side_points]))

# Draw the lines of best fit
output_image = image_rgb.copy()
cv.line(output_image, (0, int(b_left)), (3000, int(3000 * a_left + b_left)), (255, 1, 1), 5)
cv.line(output_image, (0, int(b_right)), (3000, int(3000 * a_right + b_right)), (255, 1, 1), 5)

# Show and save the final output
plt.imshow(output_image)
plt.savefig("solution.png")
plt.show()