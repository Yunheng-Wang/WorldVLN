import numpy as np
from typing import Tuple
import cv2

def resize_with_padding(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image with aspect ratio preservation and padding to target size.
    
    This function ensures no image distortion by:
    1. Calculating the minimum scale ratio to fit the image within target size
    2. Resizing the image with this ratio to preserve aspect ratio
    3. Padding with black borders to reach exact target size
    4. Centering the resized image within the padded frame
    
    Args:
        frame: Input image [H, W, C]
        target_size: Target size (height, width)
        
    Returns:
        Processed image [target_height, target_width, C]
        
    Example:
        >>> frame = np.random.randint(0, 255, (720, 640, 3), dtype=np.uint8)
        >>> resized = resize_with_padding(frame, (384, 320))
        >>> print(resized.shape)  # (384, 320, 3)
    """
    target_height, target_width = target_size
    original_height, original_width = frame.shape[:2]
    
    # Calculate scaling ratio, use the smaller ratio to ensure image fits completely
    scale_height = target_height / original_height
    scale_width = target_width / original_width
    scale = min(scale_height, scale_width)
    
    # Calculate new dimensions after scaling
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    
    # Resize with aspect ratio preservation
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Create black background with target size
    padded_frame = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
    
    # Calculate center placement position
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image at center
    padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
    
    return padded_frame