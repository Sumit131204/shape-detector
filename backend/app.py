from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import math
from rembg import remove
import uuid
import traceback
import imutils
import pandas as pd  # Add import for pandas to read Excel file
import time
import json
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Camera parameters (calibrated for accurate size measurement)
CAMERA_DISTANCE = 300.0  # mm (distance from camera to object)
IMAGE_WIDTH_PIXELS = 1280  # Standard width for processing
SENSOR_WIDTH_MM = 4.8  # Typical sensor width for smartphone cameras
FOCAL_LENGTH_MM = 35.0  # Typical focal length

# Calibration factor to improve accuracy (adjust based on testing)
# This factor compensates for systematic errors in the measurement
CALIBRATION_FACTOR = 1.02  # Slightly adjusted based on testing with known objects

# Load color data from Excel file
try:
    color_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colors.xlsx')
    color_df = pd.read_excel(color_data_path)
    
    # Print debug info
    print(f"Excel file columns: {color_df.columns.tolist()}")
    
    # Convert the data into a dictionary for faster lookup
    color_dict = {}
    for index, row in color_df.iterrows():
        try:
            # Use the correct column names found in the Excel file
            color_name = row['Color_name'] if 'Color_name' in row else f"Color_{index}"
            hex_value = row['Hex'] if 'Hex' in row else ""
            
            # The color components are in D2 (R), D1 (G), D0 (B) columns
            r = int(row['D2']) if 'D2' in row and pd.notna(row['D2']) else 0
            g = int(row['D1']) if 'D1' in row and pd.notna(row['D1']) else 0 
            b = int(row['D0']) if 'D0' in row and pd.notna(row['D0']) else 0
            
            decimal_value = (r, g, b)
            color_dict[color_name] = {'Hex': hex_value, 'Decimal': decimal_value}
        except Exception as ex:
            print(f"Error loading color row {index}: {ex}")
            continue
    
    print(f"Successfully loaded {len(color_dict)} colors from Excel file")
    
    # Print a few sample color entries for debugging
    sample_colors = list(color_dict.items())[:5]
    print("Sample colors:")
    for name, data in sample_colors:
        print(f"  {name}: {data}")
    
except Exception as e:
    print(f"Error loading color data: {str(e)}")
    # Provide basic colors as fallback
    color_dict = {
        'Red': {'Hex': '#FF0000', 'Decimal': (255, 0, 0)},
        'Green': {'Hex': '#00FF00', 'Decimal': (0, 255, 0)},
        'Blue': {'Hex': '#0000FF', 'Decimal': (0, 0, 255)},
        'Yellow': {'Hex': '#FFFF00', 'Decimal': (255, 255, 0)},
        'Orange': {'Hex': '#FFA500', 'Decimal': (255, 165, 0)},
        'Purple': {'Hex': '#800080', 'Decimal': (128, 0, 128)},
        'Brown': {'Hex': '#A52A2A', 'Decimal': (165, 42, 42)},
        'Black': {'Hex': '#000000', 'Decimal': (0, 0, 0)},
        'White': {'Hex': '#FFFFFF', 'Decimal': (255, 255, 255)},
        'Gray': {'Hex': '#808080', 'Decimal': (128, 128, 128)}
    }
    print(f"Using {len(color_dict)} basic colors as fallback")

# Function to get dominant color inside a contour
def get_avg_color(c, image):
    """
    Calculate the dominant color inside a contour using improved k-means clustering.
    
    Parameters:
    c (numpy.ndarray): Contour points
    image (numpy.ndarray): Image in BGR format
    
    Returns:
    tuple: RGB color tuple
    """
    # Create mask for the contour
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    
    # Extract only the contour pixels
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked_image[mask == 255]
    
    if len(pixels) == 0:
        return (0, 0, 0)  # Return black if no pixels found
    
    # Better color extraction using k-means clustering to find dominant colors
    pixels = pixels.reshape(-1, 3).astype(np.float32)
    
    # If we have too many pixels, sample a subset for efficiency
    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]
    
    # Use k-means clustering to find the dominant colors - try multiple clusters
    k = min(5, len(pixels) // 500 + 1)  # Dynamic k based on number of pixels
    if k < 1:
        k = 1
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    flags = cv2.KMEANS_PP_CENTERS  # Use k-means++ initialization for better results
    
    compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
    
    # Count occurrences of each cluster to find most common color
    counts = np.bincount(labels.flatten())
    
    # Find the dominant color clusters (top 2)
    dominant_idx = np.argsort(counts)[::-1][:2]
    
    # Get the most saturated color among the dominant ones
    # This helps to avoid picking greyish backgrounds
    dominant_colors_bgr = centers[dominant_idx]
    
    # Convert BGR to HSV for saturation comparison
    dominant_colors_hsv = []
    for color in dominant_colors_bgr:
        bgr_color = np.uint8([[color]])
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
        dominant_colors_hsv.append((hsv_color, color))
    
    # Sort by saturation (higher is better)
    dominant_colors_hsv.sort(key=lambda x: x[0][1], reverse=True)
    
    # Get the most saturated color that's not too dark
    for hsv, bgr in dominant_colors_hsv:
        # Skip very dark colors (low value in HSV)
        if hsv[2] > 30:  # Minimum value threshold
            dominant_color = bgr
            break
    else:
        # If all are too dark, take the brightest one
        dominant_colors_hsv.sort(key=lambda x: x[0][2], reverse=True)
        dominant_color = dominant_colors_hsv[0][1]
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    return tuple(map(int, dominant_color[::-1]))  # Reverse BGR to RGB

# Function to find the closest color name
def closest_color(requested_color):
    """
    Find the closest color name for a given RGB value using improved HSV color space comparison.
    
    Parameters:
    requested_color (tuple): RGB color tuple
    
    Returns:
    str: Closest matching color name
    """
    if not color_dict:
        return "Color data unavailable"
        
    min_distance = float("inf")
    closest_match = "Unknown"
    
    # Convert requested color to RGB values
    r, g, b = requested_color
    
    # First, handle special cases using RGB values directly
    # Check for grayscale
    avg = (r + g + b) / 3
    max_diff = max(abs(r - avg), abs(g - avg), abs(b - avg))
    
    # If color is very close to grayscale
    if max_diff < 20:
        if avg < 40:
            return "black"
        elif avg < 85:
            return "dark gray"
        elif avg < 170:
            return "gray"
        elif avg < 240:
            return "light gray"
        else:
            return "white"
    
    # Check for very saturated basic colors
    if r > 220 and g < 60 and b < 60:
        return "red"
    if r < 60 and g > 220 and b < 60:
        return "green"
    if r < 60 and g < 60 and b > 220:
        return "blue"
    if r > 220 and g > 220 and b < 60:
        return "yellow"
    
    # Convert to HSV (better for color matching)
    rgb_array = np.uint8([[[r, g, b]]])
    hsv_requested = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)[0][0]
    h_requested, s_requested, v_requested = hsv_requested
    
    # Create weight factors for HSV based on perceptual importance
    # Hue is most important for color name, then saturation, then value
    if s_requested > 30:  # If color has meaningful saturation
        h_weight = 6.0
        s_weight = 2.0
        v_weight = 1.0
    else:  # For low saturation, value becomes more important
        h_weight = 1.0
        s_weight = 2.0
        v_weight = 5.0
        
    # Normalize weights for better numerical stability
    total_weight = h_weight + s_weight + v_weight
    h_weight /= total_weight
    s_weight /= total_weight
    v_weight /= total_weight
    
    # List to track top 3 closest colors for debugging
    top_matches = []
    
    # Go through all colors and find the closest match
    for color_name, data in color_dict.items():
        try:
            # Get RGB values
            r_c, g_c, b_c = data["Decimal"]
            
            # Convert reference color to HSV
            rgb_ref = np.uint8([[[r_c, g_c, b_c]]])
            hsv_ref = cv2.cvtColor(rgb_ref, cv2.COLOR_RGB2HSV)[0][0]
            h_ref, s_ref, v_ref = hsv_ref
            
            # Calculate normalized weighted distance in HSV space
            # Handle hue's circular nature (0-180 in OpenCV)
            h_diff = min(abs(h_ref - h_requested), 180 - abs(h_ref - h_requested)) / 180.0
            s_diff = abs(s_ref - s_requested) / 255.0
            v_diff = abs(v_ref - v_requested) / 255.0
            
            # Calculate weighted distance using squared differences
            distance = (h_weight * h_diff)**2 + (s_weight * s_diff)**2 + (v_weight * v_diff)**2
            
            # Track top matches
            top_matches.append((color_name, distance, (r_c, g_c, b_c)))
            if len(top_matches) > 3:
                top_matches.sort(key=lambda x: x[1])
                top_matches = top_matches[:3]
            
            if distance < min_distance:
                min_distance = distance
                closest_match = color_name
                
        except Exception as e:
            # Skip this color if there's an error
            continue
    
    # Print debugging info for top matches
    print("Top color matches:")
    for name, dist, rgb in sorted(top_matches, key=lambda x: x[1]):
        print(f"  {name}: RGB{rgb}, distance: {dist:.4f}")
        
    print(f"Selected color: {closest_match}, RGB: {requested_color}")
    
    return closest_match

# Function to remove background
def remove_background(input_path, output_path):
    """Removes background from the input image and saves the output."""
    try:
        input_img = Image.open(input_path)
        output_img = remove(input_img)
        
        # Convert RGBA to RGB if saving as JPEG
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            bg = Image.new("RGB", output_img.size, (255, 255, 255))
            bg.paste(output_img, mask=output_img.split()[3])  # Use alpha channel as mask
            bg.save(output_path)
        else:
            output_img.save(output_path)
        
        return output_path
    except Exception as e:
        print(f"Error removing background: {str(e)}")
        # If background removal fails, copy the original image
        input_img = Image.open(input_path)
        if input_img.mode == 'RGBA':
            input_img = input_img.convert('RGB')
        input_img.save(output_path)
        return output_path

# Function to calculate Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
    pt1 (tuple): First point (x1, y1)
    pt2 (tuple): Second point (x2, y2)
    
    Returns:
    float: Euclidean distance
    """
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

# Enhanced function to calculate real-world size
def calculate_real_size(pixel_size, image_width_px):
    """
    Calculate the real-world size using multiple approaches and average them for better accuracy.
    
    Parameters:
    pixel_size (float): Size in pixels
    image_width_px (float): Width of the full image in pixels
    
    Returns:
    float: Size in millimeters
    """
    # Method 1: Using sensor width ratio
    size_mm_1 = (pixel_size * SENSOR_WIDTH_MM) / IMAGE_WIDTH_PIXELS
    real_size_mm_1 = (CAMERA_DISTANCE * size_mm_1) / FOCAL_LENGTH_MM
    
    # Method 2: Direct calculation using camera parameters
    real_size_mm_2 = (pixel_size * SENSOR_WIDTH_MM * CAMERA_DISTANCE) / (image_width_px * FOCAL_LENGTH_MM)
    
    # Method 3: Using the pinhole camera model with focal length in pixels
    focal_length_px = (image_width_px * FOCAL_LENGTH_MM) / SENSOR_WIDTH_MM
    real_size_mm_3 = (pixel_size * CAMERA_DISTANCE) / focal_length_px
    
    # Weighted average of the three methods (we give higher weight to more stable methods)
    real_size_mm = (0.4 * real_size_mm_1 + 0.4 * real_size_mm_2 + 0.2 * real_size_mm_3)
    
    # Apply calibration factor
    return real_size_mm * CALIBRATION_FACTOR

# Function to calculate accurate sizes for different shapes
def calculate_shape_size(shape, contour, image_width_px):
    """Calculate real-world size of a shape from its contour."""
    # Calculate area in pixels
    pixel_area = cv2.contourArea(contour)
    
    # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    
    # Initialize result dictionary
    measurements = {}
    
    # Calculate shape-specific measurements
    if shape == "circle":
        # For circles, use area formula πr²
        radius_px = perimeter / (2 * math.pi)
        radius_mm = calculate_real_size(radius_px, image_width_px)
        diameter_mm = radius_mm * 2
        area_mm2 = math.pi * (radius_mm ** 2)
        
        # Add circle-specific measurements
        measurements['radius_mm'] = round(radius_mm, 1)
        measurements['diameter_mm'] = round(diameter_mm, 1)
        measurements['dimension_text'] = f"D: {diameter_mm:.1f}mm"
        
    elif shape in ["rectangle", "square"]:
        # Get rotated rectangle for better measurements
        rect = cv2.minAreaRect(contour)
        width_px, height_px = rect[1]
        
        # Convert to real-world dimensions
        width_mm = calculate_real_size(width_px, image_width_px)
        height_mm = calculate_real_size(height_px, image_width_px)
        
        # Ensure width is always the longer dimension for consistency
        if width_mm < height_mm and shape == "rectangle":
            width_mm, height_mm = height_mm, width_mm
            
        # Add rectangle-specific measurements
        measurements['width_mm'] = round(width_mm, 1)
        measurements['height_mm'] = round(height_mm, 1)
        
        if shape == "square":
            # For squares, average the width and height for a more accurate side length
            side_mm = (width_mm + height_mm) / 2
            measurements['side_mm'] = round(side_mm, 1)
            measurements['dimension_text'] = f"Side: {side_mm:.1f}mm"
            area_mm2 = side_mm ** 2
        else:
            measurements['dimension_text'] = f"W: {width_mm:.1f}mm, H: {height_mm:.1f}mm"
            area_mm2 = width_mm * height_mm
            
    elif shape == "triangle":
        # For triangles, use Heron's formula with the side lengths
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 3:
            # Calculate the lengths of the three sides in pixels
            side_lengths_px = []
            for i in range(3):
                pt1 = approx[i][0]
                pt2 = approx[(i+1)%3][0]
                side_px = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                side_lengths_px.append(side_px)
            
            # Convert to mm
            side_lengths_mm = [calculate_real_size(side_px, image_width_px) for side_px in side_lengths_px]
            
            # Add triangle-specific measurements
            measurements['sides_mm'] = [round(side, 1) for side in side_lengths_mm]
            
            # Calculate area using Heron's formula
            a, b, c = side_lengths_mm
            s = (a + b + c) / 2  # semi-perimeter
            area_mm2 = math.sqrt(s * (s - a) * (s - b) * (s - c))
            
            measurements['dimension_text'] = f"Sides: {side_lengths_mm[0]:.1f}, {side_lengths_mm[1]:.1f}, {side_lengths_mm[2]:.1f}mm"
        else:
            # Fallback to simple area calculation
            area_mm2 = (pixel_area * (SENSOR_WIDTH_MM ** 2)) / (image_width_px ** 2) * ((CAMERA_DISTANCE / FOCAL_LENGTH_MM) ** 2)
            measurements['dimension_text'] = f"Area: {area_mm2:.1f}mm²"
    else:
        # For other shapes, use the contour area directly
        area_mm2 = (pixel_area * (SENSOR_WIDTH_MM ** 2)) / (image_width_px ** 2) * ((CAMERA_DISTANCE / FOCAL_LENGTH_MM) ** 2)
        measurements['dimension_text'] = f"Area: {area_mm2:.1f}mm²"
    
    # Add common measurements to all shapes
    measurements['area_pixels'] = round(pixel_area, 1)
    measurements['area_mm2'] = round(area_mm2, 1)
    measurements['perimeter_px'] = round(perimeter, 1)
    measurements['perimeter_mm'] = round(calculate_real_size(perimeter, image_width_px), 1)
    
    # Apply calibration factor to all measurements
    for key in measurements:
        if key not in ['dimension_text'] and isinstance(measurements[key], (int, float)):
            measurements[key] = round(measurements[key] * CALIBRATION_FACTOR, 1)
        elif key == 'sides_mm' and isinstance(measurements[key], list):
            measurements[key] = [round(side * CALIBRATION_FACTOR, 1) for side in measurements[key]]
    
    # Update dimension text with calibrated values
    if shape == "circle":
        measurements['dimension_text'] = f"D: {measurements['diameter_mm']}mm"
    elif shape == "square":
        measurements['dimension_text'] = f"Side: {measurements['side_mm']}mm"
    elif shape == "rectangle":
        measurements['dimension_text'] = f"W: {measurements['width_mm']}mm, H: {measurements['height_mm']}mm"
    elif shape == "triangle" and 'sides_mm' in measurements:
        sides = measurements['sides_mm']
        measurements['dimension_text'] = f"Sides: {sides[0]}, {sides[1]}, {sides[2]}mm"
    else:
        measurements['dimension_text'] = f"Area: {measurements['area_mm2']}mm²"
    
    return measurements

# Improved preprocessing function for more accurate contour detection
def preprocess_image_for_contours(image):
    """
    Apply advanced preprocessing to improve contour detection.
    
    Parameters:
    image (numpy.ndarray): Input image in BGR format
    
    Returns:
    numpy.ndarray: Processed binary image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filtering preserves edges better than gaussian blur
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding to account for variable lighting
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 13, 2
    )
    
    # Apply morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Apply Canny edge detection as an additional processing step
    edges = cv2.Canny(blurred, 30, 150)
    
    # Combine thresholded image and edges using bitwise OR
    combined = cv2.bitwise_or(binary, edges)
    
    # Final dilation to connect any remaining gaps in the contours
    dilated = cv2.dilate(combined, kernel, iterations=1)
    
    return dilated

# Function to find the largest contour (main shape) in the image
def find_main_shape(contours):
    """Find the main shape in the image (assumes it's the largest contour)."""
    if not contours:
        return None
    
    # Filter out small contours
    min_area = 50  # Minimum area in pixels
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        return None
    
    # Return the largest contour by area
    return max(valid_contours, key=cv2.contourArea)

# Function to smooth contours and remove noise
def smooth_contour(contour, factor=0.005):
    """Smooth a contour by approximating it with fewer points."""
    epsilon = factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # Initialize shape name and approximate the contour
        shape = "unknown"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True) # Slightly lower epsilon for better polygon detection
        
        # Calculate shape metrics
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Get contour area and bounding rect area for "fullness" ratio
        contour_area = cv2.contourArea(c)
        rect_area = w * h
        fullness = contour_area / float(rect_area) if rect_area > 0 else 0
        
        # For rotated rectangles
        min_area_rect = cv2.minAreaRect(c)
        min_rect_w, min_rect_h = min_area_rect[1]
        min_rect_area = min_rect_w * min_rect_h if min_rect_w and min_rect_h else 0
        min_rect_ratio = min(min_rect_w, min_rect_h) / max(min_rect_w, min_rect_h) if max(min_rect_w, min_rect_h) > 0 else 0
        
        # Calculate solidity - ratio of contour area to convex hull area
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / float(hull_area) if hull_area > 0 else 0
        
        # Calculate circularity (4π × area / perimeter²)
        circularity = 4 * np.pi * contour_area / (peri * peri) if peri > 0 else 0
        
        # Determine shape based on vertices and metrics
        if len(approx) == 3:
            shape = "triangle"
            
        elif len(approx) == 4:
            # Check if it's a square or rectangle
            if 0.95 <= aspect_ratio <= 1.05:
                # Square has aspect ratio close to 1
                shape = "square"
            elif 0.95 <= min_rect_ratio <= 1.05:
                # Rotated square
                shape = "square"
            else:
                shape = "rectangle"
                
        elif len(approx) == 5:
            shape = "pentagon"
            
        elif len(approx) == 6:
            shape = "hexagon"
            
        elif len(approx) > 6:
            # Use circularity and solidity to distinguish circles and ellipses
            if circularity > 0.85 and solidity > 0.9:
                # Check if it's more circular or elliptical
                if min_rect_ratio > 0.9:
                    shape = "circle"
                else:
                    shape = "ellipse"
            else:
                # Fit an ellipse and check how well it fits
                if len(c) >= 5:  # Need at least 5 points to fit an ellipse
                    ellipse = cv2.fitEllipse(c)
                    ellipse_area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
                    area_ratio = contour_area / ellipse_area if ellipse_area > 0 else 0
                    if 0.9 < area_ratio < 1.1:
                        if min_rect_ratio > 0.9:  # Almost equal axes = circle
                            shape = "circle"
                        else:
                            shape = "ellipse"
                    else:
                        shape = "irregular"
                else:
                    shape = "irregular"
        
        return shape

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': '/static/uploads/' + filename
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/detect-shape', methods=['POST'])
def detect_shape():
    # Get filename from request data
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
    
    filename = data['filename']
    print(f"[DEBUG] Shape detection requested for file: {filename}")
    
    try:
        # Original image path
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Generate output paths for background-removed image
        bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{filename}")
        if not bg_removed_path.lower().endswith('.png'):
            bg_removed_path = bg_removed_path.rsplit('.', 1)[0] + '.png'
        
        # Check if original file exists
        if not os.path.exists(original_path):
            print(f"[ERROR] Original file not found: {original_path}")
            return jsonify({'error': f'File not found: {original_path}'}), 404
        
        # Remove background and save as PNG
        print(f"[DEBUG] Removing background from {original_path}")
        remove_background(original_path, bg_removed_path)
        
        # Check if background removal was successful
        if not os.path.exists(bg_removed_path):
            print(f"[ERROR] Failed to remove background: {bg_removed_path}")
            return jsonify({'error': 'Failed to process image'}), 500
            
        # Read the background-removed image
        img = cv2.imread(bg_removed_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"[ERROR] Failed to read image after background removal: {bg_removed_path}")
            return jsonify({'error': 'Failed to read processed image'}), 500
        
        print(f"[DEBUG] Successfully loaded image with shape: {img.shape}")
        
        # Resize image while maintaining aspect ratio
        max_dimension = 1000  # Increase from 800 to 1000 for better accuracy
        h, w = img.shape[:2]
        
        # Calculate new dimensions
        if h > w:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))
        else:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))
            
        # Resize image
        print(f"[DEBUG] Resizing image from {w}x{h} to {new_w}x{new_h}")
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Create a visualization image that we'll draw on
        # Convert RGBA to RGB for visualization if necessary
        if len(img_resized.shape) > 2 and img_resized.shape[2] == 4:  # RGBA
            print("[DEBUG] Converting RGBA image to RGB for visualization")
            # Create a white background
            viz_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
            
            # Get the RGB and alpha channels
            rgb = img_resized[:,:,:3]
            alpha = img_resized[:,:,3]
            
            # Create a mask from the alpha channel
            alpha_mask = alpha > 0
            
            # Copy RGB values from the original image to the viz image where alpha > 0
            viz_img[alpha_mask] = rgb[alpha_mask]
        else:
            print("[DEBUG] Image is already in RGB format")
            viz_img = img_resized.copy()
        
        # Create a preprocessed image for contour detection
        print("[DEBUG] Preprocessing image for contour detection")
        processed = preprocess_image_for_contours(img_resized)
        
        # Find contours in the processed image
        print("[DEBUG] Finding contours")
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[DEBUG] Found {len(contours)} contours")
        
        # Find the main shape (largest contour)
        main_contour = find_main_shape(contours)
        
        if main_contour is None:
            print("[ERROR] No main shape detected")
            return jsonify({'error': 'No shape detected'}), 400
            
        # Smooth the contour to reduce noise
        print("[DEBUG] Smoothing contour")
        smoothed_contour = smooth_contour(main_contour)
        
        # Initialize shape detector
        sd = ShapeDetector()
        
        # Detect the shape
        shape = sd.detect(smoothed_contour)
        print(f"[DEBUG] Detected shape: {shape}")
        
        # Calculate shape dimensions
        print("[DEBUG] Calculating shape dimensions")
        measurements = calculate_shape_size(shape, smoothed_contour, new_w)
        print(f"[DEBUG] Measurements: {measurements}")
        
        # Draw the contour on the visualization image
        cv2.drawContours(viz_img, [smoothed_contour], -1, (0, 255, 0), 2)
        
        # Determine the center of the contour
        M = cv2.moments(smoothed_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = new_w // 2, new_h // 2
        
        # Add shape label
        print(f"[DEBUG] Adding visual elements to output image")
        cv2.putText(viz_img, shape.capitalize(), (cx - 20, cy - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add dimension text
        cv2.putText(viz_img, measurements['dimension_text'], (cx - 40, cy + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw shape-specific visualizations
        if shape == "circle":
            # Draw circle center and radius
            radius_px = int(measurements['radius_mm'] / calculate_real_size(1, new_w))
            cv2.circle(viz_img, (cx, cy), 3, (0, 0, 255), -1)  # Center point
            cv2.circle(viz_img, (cx, cy), radius_px, (255, 0, 0), 2)  # Circle outline
            cv2.line(viz_img, (cx, cy), (cx + radius_px, cy), (255, 0, 0), 2)  # Radius line
            
        elif shape in ["rectangle", "square"]:
            # Draw the rotated rectangle
            rect = cv2.minAreaRect(smoothed_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(viz_img, [box], 0, (255, 0, 0), 2)
            
            # Draw the width and height lines
            width = rect[1][0]
            height = rect[1][1]
            angle = rect[2]
            
            # Draw dimension lines only if they're clear enough to see (not for tiny objects)
            if width > 20 and height > 20:
                # Find the top-left, top-right, bottom-right, and bottom-left points
                # Sort the box points by their y-coordinate (top to bottom)
                sorted_by_y = sorted(box, key=lambda p: p[1])
                top_points = sorted(sorted_by_y[:2], key=lambda p: p[0])
                bottom_points = sorted(sorted_by_y[2:], key=lambda p: p[0])
                
                top_left, top_right = top_points
                bottom_left, bottom_right = bottom_points
                
                # Draw width line (top)
                cv2.line(viz_img, tuple(top_left), tuple(top_right), (255, 0, 0), 2)
                width_text_pos = ((top_left[0] + top_right[0]) // 2, top_left[1] - 10)
                
                # Draw height line (left)
                cv2.line(viz_img, tuple(top_left), tuple(bottom_left), (255, 0, 0), 2)
                height_text_pos = (top_left[0] - 40, (top_left[1] + bottom_left[1]) // 2)
                
                # Add dimension labels
                if shape == "rectangle":
                    cv2.putText(viz_img, f"W: {measurements['width_mm']}mm", width_text_pos,
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(viz_img, f"H: {measurements['height_mm']}mm", height_text_pos,
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:  # square
                    cv2.putText(viz_img, f"Side: {measurements['side_mm']}mm", width_text_pos,
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        elif shape == "triangle":
            # Draw the triangle vertices and sides
            approx = cv2.approxPolyDP(smoothed_contour, 0.02 * cv2.arcLength(smoothed_contour, True), True)
            
            if len(approx) == 3:
                # Draw the vertices
                for point in approx:
                    cv2.circle(viz_img, tuple(point[0]), 5, (0, 0, 255), -1)
                
                # Draw the side labels if we have side measurements
                if 'sides_mm' in measurements and len(measurements['sides_mm']) == 3:
                    sides = measurements['sides_mm']
                    
                    # Calculate midpoints of each side for labels
                    for i in range(3):
                        pt1 = approx[i][0]
                        pt2 = approx[(i+1)%3][0]
                        mid_x = (pt1[0] + pt2[0]) // 2
                        mid_y = (pt1[1] + pt2[1]) // 2
                        
                        # Add side length label
                        cv2.putText(viz_img, f"{sides[i]}mm", (mid_x, mid_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save the visualization image
        print(f"[DEBUG] Saving visualization image")
        viz_path = os.path.join(app.config['UPLOAD_FOLDER'], f"viz_{filename}")
        cv2.imwrite(viz_path, viz_img)
        print(f"[DEBUG] Visualization saved to: {viz_path}")
        
        # Create a shape data object for the response
        shape_data = {
            'shape': shape,
            'area_mm2': measurements.get('area_mm2', 0),
            'dimensions': measurements['dimension_text']
        }
        
        # Add all measurements to the result
        for key, value in measurements.items():
            if key != 'dimension_text':
                shape_data[key] = value
        
        # Modify the response to match the frontend's expected format
        print(f"[DEBUG] Returning successful response with shape: {shape}")
        return jsonify({
            'success': True,
            'shapes': [shape_data],  # Frontend expects 'shapes' array
            'processedImage': f"/static/uploads/viz_{filename}"
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ERROR] Error in shape detection: {str(e)}\n{error_trace}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect-size', methods=['POST'])
def detect_size():
    data = request.json
    
    if 'filename' not in data:
        return jsonify({'error': 'Filename not provided'}), 400
    
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Process image for size detection
    try:
        # Generate output paths
        bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{filename}")
        if not bg_removed_path.lower().endswith('.png'):
            bg_removed_path = bg_removed_path.rsplit('.', 1)[0] + '.png'
        
        # Step 1: Remove background and save as PNG
        remove_background(filepath, bg_removed_path)
        
        # Step 2: Load image
        image = cv2.imread(bg_removed_path)
        if image is None:
            raise Exception("Failed to load image after background removal")
        
        # Store original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Step 3: Resize image for processing if needed
        max_dim = 1200  # Increased for better accuracy
        if max(orig_height, orig_width) > max_dim:
            scale_factor = max_dim / max(orig_height, orig_width)
            resized_width = int(orig_width * scale_factor)
            resized_height = int(orig_height * scale_factor)
            dim = (resized_width, resized_height)
            image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        else:
            # No resizing needed
            image_resized = image.copy()
            resized_width, resized_height = orig_width, orig_height
            scale_factor = 1.0
        
        # Step 4: Apply improved edge detection and preprocessing
        binary = preprocess_image_for_contours(image_resized)
        
        # Apply additional morphological operations to improve contour detection
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)
        
        # Step 5: Find contours with improved method
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Process contours
        size_image = image.copy()
        results = []
        
        for contour in contours:
            # Skip very small contours
            if cv2.contourArea(contour) < 300:  # Higher threshold for size detection
                continue
            
            # Scale contour to original size if image was resized
            if max(orig_height, orig_width) > max_dim:
                contour = contour * (orig_width / resized_width)
                contour = contour.astype(np.int32)
            
            # Use minimum area rotated rectangle for better measurements
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # Draw rectangle
            cv2.drawContours(size_image, [box], 0, (0, 255, 0), 2)
            
            # Get width and height from the rotated rectangle
            (center_x, center_y), (width_px, height_px), angle = rect
            
            # Calculate real-world dimensions using our enhanced function
            width_mm = calculate_real_size(width_px, orig_width)
            height_mm = calculate_real_size(height_px, orig_width)
            
            # Ensure length is the longer dimension
            length_mm = max(width_mm, height_mm)
            breadth_mm = min(width_mm, height_mm)
            
            # Get center
            cx, cy = np.mean(box, axis=0).astype(int)
            
            # Display measurements with improved formatting
            cv2.putText(size_image, f"L: {length_mm:.2f} mm", (cx - 60, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(size_image, f"W: {breadth_mm:.2f} mm", (cx - 60, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(size_image, f"Area: {length_mm * breadth_mm:.2f} mm²", (cx - 60, cy + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw dimension lines for visualization
            # Length line
            midpoint1 = ((box[0][0] + box[1][0]) // 2, (box[0][1] + box[1][1]) // 2)
            midpoint2 = ((box[2][0] + box[3][0]) // 2, (box[2][1] + box[3][1]) // 2)
            cv2.line(size_image, midpoint1, midpoint2, (0, 255, 255), 1)
            
            # Width line
            midpoint3 = ((box[1][0] + box[2][0]) // 2, (box[1][1] + box[2][1]) // 2)
            midpoint4 = ((box[3][0] + box[0][0]) // 2, (box[3][1] + box[0][1]) // 2)
            cv2.line(size_image, midpoint3, midpoint4, (0, 255, 255), 1)
            
            # Save size details with additional information
            results.append({
                'length_mm': float(length_mm),
                'breadth_mm': float(breadth_mm),
                'area_mm2': float(length_mm * breadth_mm),
                'angle_degrees': float(angle)
            })
        
        # Save the processed image with measurements
        output_filename = f"size_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, size_image)
        
        return jsonify({
            'success': True,
            'measurements': results,
            'processedImage': '/static/uploads/' + output_filename
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in size detection: {str(e)}\n{error_trace}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect-color', methods=['POST'])
def detect_color():
    data = request.json
    
    if 'filename' not in data:
        return jsonify({'error': 'Filename not provided'}), 400
    
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Process image for color detection
    try:
        # Generate output paths
        bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{filename}")
        if not bg_removed_path.lower().endswith('.png'):
            bg_removed_path = bg_removed_path.rsplit('.', 1)[0] + '.png'
        
        # Remove background and save as PNG
        remove_background(filepath, bg_removed_path)
        
        # Load image
        image = cv2.imread(bg_removed_path)
        if image is None:
            raise Exception("Failed to load image after background removal")
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Resize for processing while maintaining aspect ratio
        max_dim = 800
        scale_factor = min(max_dim / img_width, max_dim / img_height)
        if scale_factor < 1:
            resized_width = int(img_width * scale_factor)
            resized_height = int(img_height * scale_factor)
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            ratio = img_height / float(resized_height)
        else:
            resized = image.copy()
            ratio = 1.0
        
        # Apply preprocessing for better contour detection
        binary = preprocess_image_for_contours(resized)
        
        # Find contours
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sd = ShapeDetector()
        
        # Process results
        results = []
        output_image = image.copy()
        
        for c in cnts:
            # Filter small contours
            if cv2.contourArea(c) < 50:
                continue
                
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            
            # Scale contour back to original image size    
            c_orig = c.astype("float") * ratio
            c_orig = c_orig.astype("int")
            
            # Get center coordinates
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            
            # Detect shape
            shape = sd.detect(c)
            
            # Get dominant color of the shape
            avg_color = get_avg_color(c_orig, image)
            color_name = closest_color(avg_color)
            
            # Convert RGB to hex for frontend display
            hex_color = '#{:02x}{:02x}{:02x}'.format(avg_color[0], avg_color[1], avg_color[2])
            
            # Save contour details with shape and color information
            results.append({
                'shape': shape,
                'color': color_name,
                'rgb': avg_color,
                'hex': hex_color
            })
            
            # Draw contours and labels on the output image
            cv2.drawContours(output_image, [c_orig], -1, (0, 255, 0), 2)
            
            # Draw shape name and color name
            cv2.putText(output_image, f"{shape}", (cX, cY - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output_image, f"{color_name}", (cX, cY + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add a small rectangle with the detected color
            color_block_size = 30
            color_bg = np.zeros((color_block_size, color_block_size, 3), dtype=np.uint8)
            color_bg[:, :] = (avg_color[2], avg_color[1], avg_color[0])  # BGR for OpenCV
            
            # Overlay the color rectangle
            x_pos = cX + 60
            y_pos = cY - 15
            output_image[y_pos:y_pos+color_block_size, x_pos:x_pos+color_block_size] = color_bg
            
            # Add a black border around the color square
            cv2.rectangle(output_image, 
                         (x_pos, y_pos), 
                         (x_pos + color_block_size, y_pos + color_block_size), 
                         (0, 0, 0), 1)
        
        # Save the processed image
        output_filename = f"color_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, output_image)
        
        return jsonify({
            'success': True,
            'colors': results,
            'processedImage': '/static/uploads/' + output_filename
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in color detection: {str(e)}\n{error_trace}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect-shape-color', methods=['POST'])
def detect_shape_color():
    data = request.json
    
    if 'filename' not in data:
        return jsonify({'error': 'Filename not provided'}), 400
    
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Process image for combined shape and color detection
    try:
        # Generate output paths
        bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{filename}")
        if not bg_removed_path.lower().endswith('.png'):
            bg_removed_path = bg_removed_path.rsplit('.', 1)[0] + '.png'
        
        # Remove background and save as PNG
        remove_background(filepath, bg_removed_path)
        
        # Load image
        image = cv2.imread(bg_removed_path)
        if image is None:
            raise Exception("Failed to load image after background removal")
        
        # Get image dimensions for size calculations
        img_height, img_width = image.shape[:2]
        
        # Resize for processing while maintaining aspect ratio
        max_dim = 1000
        scale_factor = min(max_dim / img_width, max_dim / img_height)
        if scale_factor < 1:
            resized_width = int(img_width * scale_factor)
            resized_height = int(img_height * scale_factor)
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            ratio = img_height / float(resized_height)
        else:
            resized = image.copy()
            ratio = 1.0
        
        # Apply improved edge detection and preprocessing
        binary = preprocess_image_for_contours(resized)
        
        # Find contours - using RETR_EXTERNAL to get only the outermost contours
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the main shape (largest contour)
        main_contour = find_main_shape(cnts)
        
        if main_contour is None:
            return jsonify({'error': 'No shapes detected in the image'}), 400
            
        # Smooth the contour to remove noise and small deviations
        main_contour = smooth_contour(main_contour)
        
        # Scale contour back to original image size    
        if scale_factor < 1:
            main_contour = main_contour.astype("float") * ratio
            main_contour = main_contour.astype("int")
        
        # Initialize ShapeDetector
        sd = ShapeDetector()
        
        # Process results
        results = []
        
        # Create a high-quality visualization image
        # Start with a clean copy of the original
        output_image = image.copy()
        
        # Draw contour with different line thickness for better visibility
        cv2.drawContours(output_image, [main_contour], -1, (0, 255, 0), 2)
        
        # Get center coordinates using moments
        M = cv2.moments(main_contour)
        if M["m00"] > 0:
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))
            
            # Detect shape
            shape = sd.detect(main_contour)
            
            # Get dominant color of the shape with our improved method
            avg_color = get_avg_color(main_contour, image)
            color_name = closest_color(avg_color)
            
            # Calculate size measurements
            measurements = calculate_shape_size(shape, main_contour, img_width)
            
            # Convert RGB to hex for frontend display
            hex_color = '#{:02x}{:02x}{:02x}'.format(avg_color[0], avg_color[1], avg_color[2])
            
            # Create result object with all information
            result = {
                'shape': shape,
                'color': color_name,
                'rgb': avg_color,
                'hex': hex_color,
                'dimensions': measurements['dimension_text']
            }
            
            # Add all measurements to the result
            for key, value in measurements.items():
                if key != 'dimension_text':
                    result[key] = value
            
            results.append(result)
            
            # Create semi-transparent overlay for better text visibility
            overlay = output_image.copy()
            text_bg_color = (0, 0, 0)
            
            # Add text with improved positioning and better contrast
            text_color = (255, 255, 255)  # White text
            
            # Create a label with shape name
            label = shape.capitalize()
            
            # Calculate label positions based on shape size
            bbox = cv2.boundingRect(main_contour)
            bbox_width, bbox_height = bbox[2], bbox[3]
            
            # Position the text based on object dimensions
            if bbox_width > 100 and bbox_height > 100:
                # For larger shapes, put text inside
                label_x = cX - len(label)*5
                label_y = cY - 30
                dim_x = cX - len(measurements['dimension_text'])*4
                dim_y = cY + 10
            else:
                # For smaller shapes, position text below
                label_x = cX - len(label)*5
                label_y = cY + bbox_height//2 + 30
                dim_x = cX - len(measurements['dimension_text'])*4
                dim_y = label_y + 30
            
            # Draw text background for better visibility
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(overlay, 
                         (label_x - 5, label_y - text_size[1] - 5), 
                         (label_x + text_size[0] + 5, label_y + 5), 
                         text_bg_color, -1)
            
            dim_text_size = cv2.getTextSize(measurements['dimension_text'], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(overlay, 
                         (dim_x - 5, dim_y - dim_text_size[1] - 5), 
                         (dim_x + dim_text_size[0] + 5, dim_y + 5), 
                         text_bg_color, -1)
            
            # Apply transparency to the overlay
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)
            
            # Draw shape label
            cv2.putText(output_image, label, (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            # Draw dimension text
            cv2.putText(output_image, measurements['dimension_text'], (dim_x, dim_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Draw dimension visualization on the object
            if shape == "circle":
                # Draw diameter line for circles
                radius_px = measurements.get('radius_mm', 0) / calculate_real_size(1, img_width)
                angle_rad = math.pi / 4  # 45 degrees
                start_x = int(cX - radius_px * math.cos(angle_rad))
                start_y = int(cY - radius_px * math.sin(angle_rad))
                end_x = int(cX + radius_px * math.cos(angle_rad))
                end_y = int(cY + radius_px * math.sin(angle_rad))
                
                cv2.line(output_image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
                
                # Draw radius
                cv2.line(output_image, (cX, cY), (end_x, end_y), (0, 200, 255), 1)
                
            elif shape in ["square", "rectangle"]:
                # For squares and rectangles, draw the minimum area rectangle
                rect = cv2.minAreaRect(main_contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                
                # Draw diagonals for visualization
                cv2.line(output_image, tuple(box[0]), tuple(box[2]), (0, 255, 255), 1)
                cv2.line(output_image, tuple(box[1]), tuple(box[3]), (0, 255, 255), 1)
                
                # Draw midpoints on each side
                for i in range(4):
                    mid_x = (box[i][0] + box[(i+1)%4][0]) // 2
                    mid_y = (box[i][1] + box[(i+1)%4][1]) // 2
                    cv2.circle(output_image, (mid_x, mid_y), 3, (0, 255, 255), -1)
        
        # Save the processed image
        output_filename = f"shape_color_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, output_image)
        
        return jsonify({
            'success': True,
            'results': results,
            'processedImage': '/static/uploads/' + output_filename
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in shape and color detection: {str(e)}\n{error_trace}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Initialize application resources
def initialize_app():
    # Ensure all required directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Check if color data file exists and is readable
    try:
        color_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'color_data.xlsx')
        if os.path.exists(color_data_path):
            print(f"Color data file found at: {color_data_path}")
        else:
            print(f"Warning: Color data file not found at: {color_data_path}")
            print("Using fallback colors for detection")
    except Exception as e:
        print(f"Error checking color data: {str(e)}")
    
    print(f"Shape detection server initialized. Upload directory: {UPLOAD_FOLDER}")
    print("Available endpoints:")
    print("  - /upload: POST endpoint for uploading images")
    print("  - /detect-shape: POST endpoint for shape and color detection")
    print("  - /static/uploads/<filename>: GET endpoint for accessing processed images")

if __name__ == '__main__':
    try:
        # Initialize application resources
        initialize_app()
        
        # Run the Flask application
        print("Starting shape detection server on http://0.0.0.0:5000")
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"Error starting the server: {str(e)}")
        traceback.print_exc() 