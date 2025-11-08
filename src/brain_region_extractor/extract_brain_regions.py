#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from brain_region_extractor.atlas import AtlasRegion, load_atlas_dictionary
from brain_region_extractor.nifti import convert_to_standard_space, is_in_standard_space, load_nifti_image
from brain_region_extractor.statistics import into_serializable
from brain_region_extractor.util import print_warning

# ruff: noqa
# extract-brain-regions --atlas-image ../atlases/mni_icbm152_nlin_sym_09c_CerebrA_nifti/mni_icbm152_CerebrA_tal_nlin_sym_09c.nii --atlas-dictionary ../atlases/mni_icbm152_nlin_sym_09c_CerebrA_nifti/CerebrA_LabelDetails.csv --scan ../../COMP5411/demo_587630_V1_t1_001.nii


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='extract_brain_regions',
        description="Extract regions information of a NIfTI file using a brain atlas.",
    )

    parser.add_argument('--atlas-dictionary',
        required=True,
        help="The brain atlas CSV dictionary.")

    parser.add_argument('--atlas-image',
        required=True,
        help="The brain atlas NIfTI image.")

    parser.add_argument('--scan',
        required=True,
        help="The brain scan NIfTI image.")

    args = parser.parse_args()

    atlas_dictionary_path = Path(args.atlas_dictionary)
    atlas_image_path      = Path(args.atlas_image)
    scan_path             = Path(args.scan)

    atlas_dictionary = load_atlas_dictionary(atlas_dictionary_path)
    atlas_image      = load_nifti_image(atlas_image_path)
    scan_image       = load_nifti_image(scan_path)

    print("Atlas regions:")

    for region in atlas_dictionary.regions:
        print(f"- {region.name} ({region.value})")

    if not is_in_standard_space(atlas_image):
        print("Resampling atlas to MNI space.")
        atlas_image = convert_to_standard_space(atlas_image, 'nearest')
    else:
        print("Atlas is already in MNI space.")

    if not is_in_standard_space(scan_image):
        print("Resampling image to MNI space.")
        scan_image = convert_to_standard_space(scan_image, 'continuous')
    else:
        print("Image is already in MNI space.")

    atlas_data = atlas_image.get_fdata()
    scan_data  = scan_image.get_fdata()

    # Dictionary to store region statistics
    regions_statistics = {}

    for region in atlas_dictionary.regions:
        print(f"Processing region '{region.name}' ({region.value})")

        # Create the mask for this region.
        region_mask = (atlas_data == region.value)

        # Check if this region exists in the atlas.
        if not np.any(region_mask):
            print_warning(f"region '{region.name}' not found in atlas image")
            continue

        # Apply mask to scan data
        region_scan_data = scan_data[region_mask]

        if len(region_scan_data) == 0:
            print_warning(f"region '{region.name}' found in atlas but no corresponding scan data")
            continue

        regions_statistics[region.name] = collect_region_statistics(region, region_mask, region_scan_data)

    print(json.dumps(into_serializable(regions_statistics), indent=4))


def collect_region_statistics(
    region: AtlasRegion,
    region_mask: np.ndarray,
    region_scan_data: np.ndarray,
) -> dict[str, Any]:
    return {
        'name': region.name,
        'value': region.value,
        'voxel_count': np.sum(region_mask),
        'mean_intensity': np.mean(region_scan_data),
        'std_intensity': np.std(region_scan_data),
        'min_intensity': np.min(region_scan_data),
        'max_intensity': np.max(region_scan_data),
        'median_intensity': np.median(region_scan_data),
    }


if __name__ == "main":
    main()
