# PerceptionChallenge2024
my submission for the perception challenge for the Wisconsin Autonomous club

## Methodology
I processed the image by converting it to HSV color space to isolate red cones. Using specific color thresholds, I created masks to capture the cones. To reduce noise, I applied morphological operations and blurred the image before detecting edges with Canny.

I then found contours, simplified them, and calculated convex hulls. Hulls with 3-10 points were kept, and I filtered them based on whether they "pointed upward" (like a cone). Bounding rectangles were drawn around the detected cones, and I used least-squares regression to fit lines to the left and right sets of cones. Finally, I plotted the lines and saved the result.

## Libraries Used
- OpenCV: Image processing and contour detection
- NumPy: Array handling and calculations
- SciPy: Line fitting with least-squares
- Matplotlib: Visualization and saving output.
