import cv2
import numpy as np
from rembg import remove
from PIL import Image
import pandas as pd
from collections import Counter

# Load the Excel file with colors
df = pd.read_excel('colour2.xlsx')

# Convert to dictionary with color families
color_dict = {}
for index, row in df.iterrows():
    color_family = row['Color_name']
    hex_value = row['Hex']
    decimal_value = (int(row['D2']), int(row['D1']), int(row['D0']))  # RGB
    color_dict[color_family] = {'Hex': hex_value, 'Decimal': decimal_value}


def closest_color(requested_color):
    min_colors = {}
    for family, data in color_dict.items():
        r_c, g_c, b_c = map(int, data['Decimal'])
        distance = ((r_c - int(requested_color[0])) ** 2 +
                    (g_c - int(requested_color[1])) ** 2 +
                    (b_c - int(requested_color[2])) ** 2)
        min_colors[distance] = family
    return min_colors[min(min_colors.keys())]


def get_color_family(rgb):
    try:
        return closest_color(rgb)
    except ValueError:
        return "Unknown"


# Load and resize the image
image = cv2.imread("ball1.jpg")
if image is None:
    print("Error: Image not found or cannot be opened.")
    exit()

scale_percent = 30
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
image_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
output_pil = remove(image_pil)
image_np = np.array(output_pil)
image_no_bg = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

gray_image = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)
_, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def detect_shape(c):
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    sides = len(approx)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    if sides == 3:
        return "Triangle"
    elif sides == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif sides == 5:
        return "Pentagon"
    else:
        if perimeter == 0:
            return "Unknown"
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if 0.7 < circularity < 1.2:
            return "Circle"
        return "Unknown"


def get_dominant_color(c, image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, (255), thickness=cv2.FILLED)

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked_image[mask == 255]
    if len(pixels) == 0:
        return (0, 0, 0)

    filtered_pixels = [tuple(p) for p in pixels if not all(val > 220 for val in p)]
    if not filtered_pixels:
        filtered_pixels = [tuple(p) for p in pixels]

    most_common_pixel = Counter(filtered_pixels).most_common(1)[0][0]
    return most_common_pixel


shape_image = image_resized.copy()
detected_shapes = []
detected_colors = []

for contour in contours:
    if cv2.contourArea(contour) > 500:
        shape = detect_shape(contour)
        color = get_dominant_color(contour, image_no_bg)
        color_rgb = tuple(reversed(color))  # Convert BGR to RGB
        color_family = get_color_family(color_rgb)

        detected_shapes.append(shape)
        detected_colors.append(color_family)

        # Only draw contours without labels on shapes
        cv2.drawContours(shape_image, [contour], -1, (0, 255, 0), 2)

# Create a header section for displaying results
# Create a black background header to display all detected shapes and colors
header_height = 50  # Adjust based on the number of detected shapes
result_image = np.zeros((header_height + shape_image.shape[0], shape_image.shape[1], 3), dtype=np.uint8)
result_image[header_height:, :] = shape_image
result_image[:header_height, :] = (240, 240, 240)  # Light gray header background

# Draw label summary in header
labels = list(zip(detected_shapes, detected_colors))
unique_labels = list(set(labels))

font_scale = 0.5
font_thickness = 1
line_height = 30
start_x = 20
start_y = 30

# Add title
cv2.putText(result_image,"", (start_x, start_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Add shape and color information
for i, (shape_name, color_name) in enumerate(unique_labels):
    text_position = (start_x + 30, start_y + i * line_height)

    color_rgb = color_dict.get(color_name, {'Decimal': (255, 255, 255)})['Decimal']
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB ➝ BGR

    circle_center = (start_x + 15, start_y + i * line_height - 5)
    cv2.circle(result_image, circle_center, 7, color_bgr, -1)
    label_text = f"{shape_name} - {color_name}"
    cv2.putText(result_image, label_text, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

# Display
cv2.namedWindow('Shape Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Shape Detection', 1000, 700)
cv2.imshow('Shape Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
