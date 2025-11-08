from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn.datasets import load_mni152_template  # type: ignore
from nilearn.image import resample_img  # type: ignore

from brain_region_extractor.util import print_error_exit


def load_nifti_image(path: Path) -> Nifti1Image:
    image = nib.load(path)  # type: ignore
    if not isinstance(image, Nifti1Image):
        print_error_exit(f"file '{path}' does not contain a NIfTI1 image")

    return image  # type: ignore


def is_in_standard_space(image: Nifti1Image) -> bool:
    mni_template = load_mni152_template()  # type: ignore
    return np.allclose(image.affine, mni_template.affine) and image.shape == mni_template.shape  # type: ignore


def convert_to_standard_space(image: Nifti1Image, interpolation: Literal['nearest', 'continuous']) -> Nifti1Image:
    mni_template = load_mni152_template()  # type: ignore
    return resample_img(
        image,
        target_affine=mni_template.affine,  # type: ignore
        target_shape=mni_template.shape,  # type: ignore
        interpolation=interpolation,
        force_resample=True,
        copy_header=True,
    )  # type: ignore
