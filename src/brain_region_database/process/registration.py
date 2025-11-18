import ants  # type: ignore
from ants import ANTsImage  # type: ignore

from brain_region_database.nifti import Interpolation  # type: ignore


def register_nifti(image: ANTsImage, reference: ANTsImage, interpolation: Interpolation) -> ANTsImage:
    match interpolation:
        case 'continuous':
            interpolator = 'linear'
        case 'nearest':
            interpolator = 'nearestNeighbor'

    registration = ants.registration(fixed=reference, moving=image, type_of_transform='SyN')  # type: ignore

    return ants.apply_transforms(  # type: ignore
        fixed=reference,
        moving=image,
        transformlist=registration['fwdtransforms'],  # type: ignore
        interpolator=interpolator,
    )
