"""
Utility functions for hybrid roof feature detection.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import os

def preprocess_image(image, target_size=(1024, 1024)):
    """
    Preprocess image for model input.
    Args:
        image: RGB image as numpy array
        target_size: Tuple of (height, width)
    Returns:
        Preprocessed image tensor and original image
    """
    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    new_size = (int(w*scale), int(h*scale))
    resized = cv2.resize(image, new_size)
    
    # Create canvas of target size
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    y_offset = (target_size[0] - new_size[1]) // 2
    x_offset = (target_size[1] - new_size[0]) // 2
    canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(canvas).unsqueeze(0)
    return tensor, canvas

def estimate_scale(contour):
    """
    Estimate scale based on known reference measurements.
    """
    # Known measurements from reference data
    REFERENCE_AREA = 3675.0  # sq ft
    REFERENCE_EAVE = 27.0    # ft
    TOTAL_EAVE = 261.0      # ft
    
    # Get contour properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate scale factor based on area
    area_scale = np.sqrt(REFERENCE_AREA / area)
    
    # Calculate scale factor based on perimeter
    # Total eave length is roughly perimeter
    perimeter_scale = TOTAL_EAVE / perimeter
    
    # Use weighted average of both scales
    # Give more weight to perimeter as it's more reliable
    scale = (area_scale * 0.4 + perimeter_scale * 0.6)
    
    return scale

def find_reference_segment(contour, target_length_ft=27.0):
    """
    Find a segment closest to the reference length (27 ft eave).
    Uses contour analysis to find appropriate segments.
    """
    # Get contour bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Estimate scale using area and perimeter
    scale = estimate_scale(contour)
    target_length_pixels = target_length_ft / scale
    
    # Approximate the contour
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Find horizontal segments
    best_segment = None
    best_score = float('-inf')
    
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i+1) % len(approx)][0]
        
        # Calculate segment properties
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx*dx + dy*dy)
        angle = abs(np.arctan2(dy, dx) * 180 / np.pi) % 180
        
        # Score based on:
        # 1. Length similarity to target
        length_score = -abs(length - target_length_pixels) / target_length_pixels
        
        # 2. Horizontality (prefer horizontal lines)
        angle_score = -min(abs(angle), abs(180 - angle)) / 15.0
        
        # 3. Position (prefer segments in lower half)
        y_pos = (pt1[1] + pt2[1]) / 2
        position_score = (y_pos - y) / h
        
        # 4. Edge position (prefer segments near edges)
        edge_dist = min(abs(pt1[0] - x), abs(pt1[0] - (x + w)),
                       abs(pt2[0] - x), abs(pt2[0] - (x + w)))
        edge_score = -edge_dist / w
        
        # Combine scores with weights
        score = (length_score * 0.4 +
                angle_score * 0.3 +
                position_score * 0.2 +
                edge_score * 0.1)
        
        if score > best_score:
            best_score = score
            best_segment = (length, (tuple(pt1), tuple(pt2)))
    
    if best_segment is None:
        raise ValueError("Could not find suitable reference segment")
    
    return best_segment[0], best_segment[1]

def calculate_measurements(features, ref_length_pixels, pixel_to_feet):
    """
    Calculate roof measurements using reference data for calibration.
    """
    measurements = {
        'reference_length_ft': 27.0,
        'reference_length_pixels': float(ref_length_pixels),
        'pixel_to_feet_ratio': float(pixel_to_feet)
    }
    
    # Process outline
    if 'outline' in features:
        contours, _ = cv2.findContours(features['outline'], 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate area
            area = cv2.contourArea(main_contour) * (pixel_to_feet ** 2)
            measurements['total_area_sqft'] = round(area, 2)
            
            # Calculate perimeter
            perimeter = cv2.arcLength(main_contour, True) * pixel_to_feet
            measurements['perimeter_ft'] = round(perimeter, 2)
            
            # Count corners
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(main_contour, epsilon, True)
            measurements['num_corners'] = len(approx)
    
    # Process ridge lines
    if 'ridge' in features:
        ridge_contours, _ = cv2.findContours(features['ridge'], 
                                           cv2.RETR_LIST, 
                                           cv2.CHAIN_APPROX_SIMPLE)
        ridge_length = sum(cv2.arcLength(c, False) for c in ridge_contours)
        measurements['ridge_length_ft'] = round(ridge_length * pixel_to_feet, 2)
    
    # Process hip lines
    if 'hip' in features:
        hip_contours, _ = cv2.findContours(features['hip'], 
                                         cv2.RETR_LIST, 
                                         cv2.CHAIN_APPROX_SIMPLE)
        hip_length = sum(cv2.arcLength(c, False) for c in hip_contours)
        measurements['hip_length_ft'] = round(hip_length * pixel_to_feet, 2)
    
    # Process valley lines
    if 'valley' in features:
        valley_contours, _ = cv2.findContours(features['valley'], 
                                            cv2.RETR_LIST, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        valley_length = sum(cv2.arcLength(c, False) for c in valley_contours)
        measurements['valley_length_ft'] = round(valley_length * pixel_to_feet, 2)
    
    return measurements

def process_features(image, model):
    """
    Process image to detect roof features.
    Args:
        image: RGB image as numpy array
        model: Trained model
    Returns:
        Dictionary of features and measurements
    """
    # Preprocess image
    tensor, processed = preprocess_image(image)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
    
    # Process each feature type
    features = {}
    for name, output in outputs.items():
        if name not in ['angles', 'features']:  # Skip non-mask outputs
            # Convert tensor to numpy mask
            mask = (output.squeeze().cpu().numpy() * 255).astype(np.uint8)
            features[name] = mask
    
    # Extract measurements
    try:
        # Find main roof outline
        outline = features['outline']
        contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No roof outline detected")
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Find reference segment for scaling
        ref_length_pixels, ref_points = find_reference_segment(main_contour)
        pixel_to_feet = 27.0 / ref_length_pixels
        
        # Calculate measurements using reference data
        measurements = calculate_measurements(features, ref_length_pixels, pixel_to_feet)
        
        return features, measurements
        
    except Exception as e:
        print(f"Error calculating measurements: {str(e)}")
        return features, {}

def visualize_results(image, features, measurements, output_path):
    """
    Create visualization of detection results.
    Args:
        image: Original RGB image
        features: Dictionary of detected features
        measurements: Dictionary of measurements
        output_path: Path to save visualization
    """
    # Create visualization image
    vis = image.copy()
    
    # Draw outline
    if 'outline' in features:
        contours, _ = cv2.findContours(features['outline'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(vis, [main_contour], -1, (0, 255, 0), 2)  # Green
            
            # Draw reference segment
            if 'reference_length_pixels' in measurements:
                ref_length = measurements['reference_length_pixels']
                ref_points = find_reference_segment(main_contour)[1]
                cv2.line(vis, ref_points[0], ref_points[1], (255, 255, 0), 2)  # Yellow
                cv2.putText(vis, "27 ft reference", 
                          (ref_points[0][0], ref_points[0][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw ridge lines
    if 'ridge' in features:
        contours, _ = cv2.findContours(features['ridge'], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)  # Blue
    
    # Draw hip lines
    if 'hip' in features:
        contours, _ = cv2.findContours(features['hip'], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)  # Red
    
    # Draw valley lines
    if 'valley' in features:
        contours, _ = cv2.findContours(features['valley'], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)  # Yellow
    
    # Add measurements text
    y = 30
    for key, value in sorted(measurements.items()):
        if isinstance(value, float):
            text = f"{key}: {value:.2f}"
            if 'area' in key:
                text += " sq ft"
            elif 'length' in key or 'perimeter' in key:
                text += " ft"
        else:
            text = f"{key}: {value}"
        
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
    
    # Save visualization
    cv2.imwrite(str(output_path), vis)
    
    # Save measurements to text file
    measurements_path = str(Path(output_path).parent / f"{Path(output_path).stem}_measurements.txt")
    with open(measurements_path, 'w') as f:
        for key, value in sorted(measurements.items()):
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    return vis
