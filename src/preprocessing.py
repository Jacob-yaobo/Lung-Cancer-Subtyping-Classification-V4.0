import os

import numpy as np
import SimpleITK as sitk

from skimage.filters import threshold_otsu

from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component

from typing import Union, Tuple, List


def remove_all_but_largest_component_from_segmentation(
    segmentation: np.ndarray,
    labels_or_regions: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]],
    background_label: int = 0,
) -> np.ndarray:
    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)
    mask_keep = remove_all_but_largest_component(mask)
    ret = np.copy(segmentation)  # do not modify the input!
    ret[mask & ~mask_keep] = background_label
    return ret


def refine_segmentation_with_otsu(
    image_sitk: sitk.Image, 
    initial_mask_sitk: sitk.Image
) -> sitk.Image:
    """
    Refines a segmentation mask using Otsu's thresholding and largest connected component analysis.
    This version accepts and returns SimpleITK images to preserve metadata.

    Args:
        image_sitk (sitk.Image): The grayscale SimpleITK image (e.g., CT or PET).
        initial_mask_sitk (sitk.Image): A SimpleITK binary mask indicating the region of interest (e.g., a bounding box).

    Returns:
        sitk.Image: The refined binary segmentation mask as a SimpleITK image.
    """
    # Convert SimpleITK images to NumPy arrays for processing
    # Note: sitk.GetArrayFromImage swaps the axis order (Z, Y, X)
    image_np = sitk.GetArrayFromImage(image_sitk)
    initial_mask_np = sitk.GetArrayFromImage(initial_mask_sitk)

    # Ensure the initial mask is boolean
    initial_mask_np = initial_mask_np.astype(bool)
    
    # Step 1: Apply Otsu's thresholding ONLY to the region of interest
    pixels_in_roi = image_np[initial_mask_np]
    
    if pixels_in_roi.size == 0:
        # Return an empty sitk image with correct metadata if the mask is empty
        refined_mask_np = np.zeros_like(image_np, dtype=np.uint8)
    else:
        otsu_thresh = threshold_otsu(pixels_in_roi)
        
        # Step 2: Create a binary mask and limit it to the initial ROI
        binary_mask = image_np > otsu_thresh
        binary_mask[~initial_mask_np] = 0
        
        # Step 3: Keep only the largest connected component
        refined_mask_np = remove_all_but_largest_component_from_segmentation(binary_mask, labels_or_regions=1)
        refined_mask_np = refined_mask_np.astype(np.uint8)

    # Convert the resulting NumPy array back to a SimpleITK image
    refined_mask_sitk = sitk.GetImageFromArray(refined_mask_np)
    
    # Copy metadata (origin, spacing, direction) from the original image
    refined_mask_sitk.CopyInformation(image_sitk)
    
    return refined_mask_sitk
