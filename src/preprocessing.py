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


# ===================================================================
# Resampling and Cropping Functions
# Author: Jacob-yaobo
# Date: 2025-09-24
# ===================================================================

def calculate_resample_template(ct_image: sitk.Image, pet_image: sitk.Image, target_spacing: list) -> sitk.Image:
    """
    Computes a template image for resampling based on the physical space of CT and PET images.

    :param ct_image: SimpleITK image of the CT scan.
    :param pet_image: SimpleITK image of the PET scan.
    :param target_spacing: The desired target spacing as a list or tuple.
    :return: A SimpleITK image to be used as a resampling template.
    """
    # Use numpy for vector operations
    import numpy as np

    # Get physical boundaries of CT image
    ct_origin = np.array(ct_image.GetOrigin())
    ct_size = np.array(ct_image.GetSize())
    ct_spacing = np.array(ct_image.GetSpacing())
    ct_end = ct_origin + (ct_size - 1) * ct_spacing

    # Get physical boundaries of PET image
    pet_origin = np.array(pet_image.GetOrigin())
    pet_size = np.array(pet_image.GetSize())
    pet_spacing = np.array(pet_image.GetSpacing())
    pet_end = pet_origin + (pet_size - 1) * pet_spacing

    # Determine the combined bounding box in physical space
    combined_origin = np.minimum(ct_origin, pet_origin)
    combined_end = np.maximum(ct_end, pet_end)

    # Calculate the size of the new template image
    new_size = np.ceil((combined_end - combined_origin) / target_spacing).astype(int)

    # Create the template image
    template_image = sitk.Image([int(s) for s in new_size], sitk.sitkFloat64)
    template_image.SetOrigin(combined_origin.tolist())
    template_image.SetSpacing(target_spacing)
    # Assume both images have the same orientation
    template_image.SetDirection(ct_image.GetDirection())

    return template_image


def resample_image(image: sitk.Image, 
                   template_image: sitk.Image, 
                   default_value: float, 
                   interpolator: int = sitk.sitkLinear) -> sitk.Image:
    """
    Resamples an image to match the grid of a template image with a specific default value.

    :param image: The SimpleITK image to resample.
    :param template_image: A SimpleITK image defining the target grid.
    :param default_value: The pixel value to use for regions outside the original image.
    :param interpolator: The interpolator to use (e.g., sitk.sitkLinear, sitk.sitkNearestNeighbor).
    :return: The resampled SimpleITK image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(template_image)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value) 

    return resampler.Execute(image)


# Crop functions
def remove_small_objects(mask: sitk.Image, min_size: int) -> sitk.Image:
    """
    Removes small connected components (noise) from a binary mask.
    This implementation is compatible with older SimpleITK versions.

    :param mask: The binary SimpleITK mask image (foreground = 1, background = 0).
    :param min_size: The minimum number of pixels for an object to be kept.
    :return: A new mask image with small objects removed.
    """
    # Ensure the mask is of an unsigned integer type
    if mask.GetPixelID() not in [sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkUInt32, sitk.sitkUInt64]:
        mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Find all connected components
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_mask = cc_filter.Execute(mask)
    
    # Get statistics for each label
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(labeled_mask)
    
    # Get all labels (excluding background 0)
    labels = label_stats.GetLabels()
    if 0 in labels:
        labels.remove(0)
        
    # Find all labels of small objects to remove
    labels_to_remove = [label for label in labels if label_stats.GetNumberOfPixels(label) < min_size]
            
    # If there are small objects, change their labels to 0 (background)
    if labels_to_remove:
        change_label_filter = sitk.ChangeLabelImageFilter()
        change_label_filter.SetChangeMap({l: 0 for l in labels_to_remove})
        relabeled_mask = change_label_filter.Execute(labeled_mask)
    else:
        relabeled_mask = labeled_mask

    # Convert the result back to a binary image
    binary_mask = sitk.BinaryThreshold(relabeled_mask, lowerThreshold=1, upperThreshold=4294967295, insideValue=1, outsideValue=0)
    
    return binary_mask


def get_bounding_box(mask: sitk.Image) -> tuple:
    """
    Calculates the bounding box of all foreground regions in a binary mask.

    :param mask: The binary SimpleITK mask image.
    :return: A tuple representing the bounding box (x_start, y_start, z_start, x_size, y_size, z_size).
    """
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask)
    
    if 1 not in label_stats.GetLabels():
        raise ValueError("No foreground objects found in the mask to calculate bounding box.")
    
    # Get the bounding box for the foreground label 1
    return label_stats.GetBoundingBox(1)


def crop_to_bounding_box(image: sitk.Image, bounding_box: tuple) -> sitk.Image:
    """
    Crops an image to a given bounding box.

    :param image: The SimpleITK image to crop.
    :param bounding_box: A tuple (x_start, y_start, z_start, x_size, y_size, z_size).
    :return: The cropped SimpleITK image.
    """
    start_index = bounding_box[0:3]
    size = bounding_box[3:6]
    return sitk.RegionOfInterest(image, size, start_index)
