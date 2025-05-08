import cv2
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd
import os

class EnhancedColorDetector:
    def __init__(self):
        # Load the Excel file with colors
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            excel_path = os.path.join(script_dir, 'colour2.xlsx')
            df = pd.read_excel(excel_path)
            
            # Convert to dictionary with color families
            self.color_dict = {}
            for index, row in df.iterrows():
                color_family = row['Color_name']
                hex_value = row['Hex']
                decimal_value = (int(row['D2']), int(row['D1']), int(row['D0']))  # RGB
                self.color_dict[color_family] = {'Hex': hex_value, 'Decimal': decimal_value}
                
            print(f"Successfully loaded {len(self.color_dict)} colors from colour2.xlsx")
        except Exception as e:
            print(f"Error loading color data from colour2.xlsx: {str(e)}")
            # Provide basic colors as fallback
            self.color_dict = {
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
            print(f"Using {len(self.color_dict)} basic colors as fallback")

    def closest_color(self, requested_color):
        """
        Find the closest color name for a given RGB value.
        
        Parameters:
        requested_color (tuple): RGB color tuple
        
        Returns:
        str: Closest matching color name
        """
        min_colors = {}
        for family, data in self.color_dict.items():
            r_c, g_c, b_c = map(int, data['Decimal'])
            distance = ((r_c - int(requested_color[0])) ** 2 +
                      (g_c - int(requested_color[1])) ** 2 +
                      (b_c - int(requested_color[2])) ** 2)
            min_colors[distance] = family
        return min_colors[min(min_colors.keys())]

    def get_color_family(self, rgb):
        """
        Get the color family for a given RGB value.
        
        Parameters:
        rgb (tuple): RGB color tuple
        
        Returns:
        str: Color family name
        """
        try:
            return self.closest_color(rgb)
        except ValueError:
            return "Unknown"

    def get_dominant_color(self, contour, image):
        """
        Get the dominant color inside a contour.
        
        Parameters:
        contour (numpy.ndarray): Contour points
        image (numpy.ndarray): Image in BGR format
        
        Returns:
        tuple: BGR color tuple
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        pixels = masked_image[mask == 255]
        if len(pixels) == 0:
            return (0, 0, 0)

        filtered_pixels = [tuple(p) for p in pixels if not all(val > 220 for val in p)]
        if not filtered_pixels:
            filtered_pixels = [tuple(p) for p in pixels]

        most_common_pixel = Counter(filtered_pixels).most_common(1)[0][0]
        return most_common_pixel

    def detect_colors(self, image_path, contours=None):
        """
        Detect colors in an image, either for the whole image or for specific contours.
        
        Parameters:
        image_path (str): Path to the image
        contours (list, optional): List of contours to detect colors for
        
        Returns:
        tuple: (processed image, list of detected colors)
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Error: Image not found or cannot be opened.")
            
        # If no contours are provided, process the whole image to find contours
        if contours is None:
            # Convert to PIL Image for background removal
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Remove background
            try:
                from rembg import remove
                output_pil = remove(image_pil)
                image_np = np.array(output_pil)
                image_no_bg = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            except Exception as e:
                print(f"Background removal failed: {e}. Using original image.")
                image_no_bg = cv2.cvtColor(image_pil, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale and threshold
            gray_image = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)
            _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create output image
        result_image = image.copy()
        detected_colors = []
        
        # Create a header section for displaying results
        header_height = 50
        result_with_header = np.zeros((header_height + result_image.shape[0], result_image.shape[1], 3), dtype=np.uint8)
        result_with_header[header_height:, :] = result_image
        result_with_header[:header_height, :] = (240, 240, 240)  # Light gray header background
        
        # Process each contour
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 500:
                # Get dominant color (in BGR)
                dominant_color_bgr = self.get_dominant_color(contour, image_no_bg if 'image_no_bg' in locals() else image)
                # Convert BGR to RGB
                dominant_color_rgb = tuple(reversed(dominant_color_bgr))
                # Get color family
                color_family = self.get_color_family(dominant_color_rgb)
                # Convert RGB to HEX
                hex_color = '#{:02x}{:02x}{:02x}'.format(dominant_color_rgb[0], dominant_color_rgb[1], dominant_color_rgb[2])
                
                # Draw contour on the result image
                cv2.drawContours(result_with_header[header_height:, :], [contour], -1, (0, 255, 0), 2)
                
                # Add detected color info to the list
                detected_colors.append({
                    'color': color_family,
                    'rgb': dominant_color_rgb,
                    'hex': hex_color
                })
        
        # Draw color information in the header
        unique_colors = {}
        for color_info in detected_colors:
            color_name = color_info['color']
            if color_name not in unique_colors:
                unique_colors[color_name] = color_info['hex']
        
        font_scale = 0.5
        font_thickness = 1
        line_height = 30
        start_x = 20
        start_y = 30
        
        # Add color information to header
        for i, (color_name, hex_value) in enumerate(unique_colors.items()):
            text_position = (start_x + 30, start_y + i * line_height - 15 if i > 0 else start_y)
            
            # Get RGB values from hex
            rgb = tuple(int(hex_value.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            bgr = (rgb[2], rgb[1], rgb[0])  # RGB to BGR
            
            # Draw color circle
            circle_center = (start_x + 15, start_y + i * line_height - 5 if i > 0 else start_y - 5)
            cv2.circle(result_with_header, circle_center, 7, bgr, -1)
            
            # Draw color name
            label_text = f"{color_name}"
            cv2.putText(result_with_header, label_text, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        return result_with_header, detected_colors

# Usage example
if __name__ == "__main__":
    color_detector = EnhancedColorDetector()
    result_image, colors = color_detector.detect_colors("test_image.jpg")
    
    # Display results
    print("Detected Colors:")
    for color in colors:
        print(f"Color: {color['color']}, RGB: {color['rgb']}, HEX: {color['hex']}")
    
    # Show the image
    cv2.imshow("Color Detection Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 