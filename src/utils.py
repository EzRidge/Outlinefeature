"""
Utility functions for post-processing and measurement calculations.
"""

import cv2
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

def extract_contours(mask, min_area=100):
    """
    Extract contours from segmentation mask.
    
    Args:
        mask (np.ndarray): Binary segmentation mask
        min_area (int): Minimum contour area to keep
        
    Returns:
        list: List of contours as numpy arrays
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter small contours
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def simplify_polygon(contour, epsilon_factor=0.02):
    """
    Simplify polygon contour using Douglas-Peucker algorithm.
    
    Args:
        contour (np.ndarray): Contour points
        epsilon_factor (float): Approximation accuracy factor
        
    Returns:
        np.ndarray: Simplified contour points
    """
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def get_roof_area(contour, pixels_per_meter=10):
    """
    Calculate roof area from contour.
    
    Args:
        contour (np.ndarray): Roof contour points
        pixels_per_meter (float): Pixel to meter conversion factor
        
    Returns:
        float: Roof area in square meters
    """
    area_pixels = cv2.contourArea(contour)
    return area_pixels / (pixels_per_meter ** 2)

def get_roof_perimeter(contour, pixels_per_meter=10):
    """
    Calculate roof perimeter from contour.
    
    Args:
        contour (np.ndarray): Roof contour points
        pixels_per_meter (float): Pixel to meter conversion factor
        
    Returns:
        float: Roof perimeter in meters
    """
    perimeter_pixels = cv2.arcLength(contour, True)
    return perimeter_pixels / pixels_per_meter

def detect_ridge_lines(mask, min_length=20):
    """
    Detect ridge lines from ridge segmentation mask.
    
    Args:
        mask (np.ndarray): Ridge line segmentation mask
        min_length (int): Minimum line length to keep
        
    Returns:
        list: List of detected lines as (x1,y1,x2,y2) tuples
    """
    edges = cv2.Canny(mask.astype(np.uint8), 50, 150)
    lines = cv2.HoughLinesP(
        edges, 
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=10
    )
    
    if lines is None:
        return []
    
    return [line[0] for line in lines]

def get_roof_pitch(ridge_lines, eave_lines):
    """
    Estimate roof pitch from ridge and eave lines.
    
    Args:
        ridge_lines (list): List of ridge lines
        eave_lines (list): List of eave lines
        
    Returns:
        float: Estimated roof pitch in degrees
    """
    if not ridge_lines or not eave_lines:
        return None
    
    # Convert to LineString objects
    ridge = LineString(ridge_lines[0])
    eave = LineString(eave_lines[0])
    
    # Get angle between lines
    angle = abs(np.arctan2(
        ridge.coords[1][1] - ridge.coords[0][1],
        ridge.coords[1][0] - ridge.coords[0][0]
    ) - np.arctan2(
        eave.coords[1][1] - eave.coords[0][1],
        eave.coords[1][0] - eave.coords[0][0]
    ))
    
    return np.degrees(angle)

def merge_close_polygons(polygons, distance_threshold=10):
    """
    Merge polygons that are within a certain distance of each other.
    
    Args:
        polygons (list): List of polygon contours
        distance_threshold (float): Distance threshold for merging
        
    Returns:
        list: List of merged polygon contours
    """
    # Convert contours to shapely polygons
    shapely_polygons = [Polygon(cnt.reshape(-1, 2)) for cnt in polygons]
    
    # Buffer polygons and merge overlapping ones
    buffered = [p.buffer(distance_threshold) for p in shapely_polygons]
    merged = unary_union(buffered).buffer(-distance_threshold)
    
    # Convert back to contours
    if merged.geom_type == 'Polygon':
        coords = np.array(merged.exterior.coords[:-1], dtype=np.int32)
        return [coords.reshape(-1, 1, 2)]
    else:
        contours = []
        for geom in merged.geoms:
            coords = np.array(geom.exterior.coords[:-1], dtype=np.int32)
            contours.append(coords.reshape(-1, 1, 2))
        return contours

def calculate_measurements(roof_mask, ridge_mask, eave_mask, pixels_per_meter=10):
    """
    Calculate comprehensive roof measurements from segmentation masks.
    
    Args:
        roof_mask (np.ndarray): Binary mask of roof outline
        ridge_mask (np.ndarray): Binary mask of ridge lines
        eave_mask (np.ndarray): Binary mask of eave lines
        pixels_per_meter (float): Pixel to meter conversion factor
        
    Returns:
        dict: Dictionary containing roof measurements
    """
    # Extract contours
    roof_contours = extract_contours(roof_mask)
    if not roof_contours:
        return None
    
    # Get main roof contour
    main_contour = max(roof_contours, key=cv2.contourArea)
    
    # Simplify polygon
    simplified_contour = simplify_polygon(main_contour)
    
    # Detect lines
    ridge_lines = detect_ridge_lines(ridge_mask)
    eave_lines = detect_ridge_lines(eave_mask)
    
    # Calculate measurements
    measurements = {
        'area': get_roof_area(simplified_contour, pixels_per_meter),
        'perimeter': get_roof_perimeter(simplified_contour, pixels_per_meter),
        'pitch': get_roof_pitch(ridge_lines, eave_lines) if ridge_lines and eave_lines else None,
        'num_facets': len(roof_contours),
        'contour_points': simplified_contour.reshape(-1, 2).tolist()
    }
    
    return measurements
