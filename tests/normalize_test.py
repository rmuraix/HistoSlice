import numpy as np
import pytest

import histoslice.functional as F
from histoslice import SlideReader
from histoslice.utils import MachenkoStainNormalizer, VahadaneStainNormalizer

from ._utils import IMAGE, SLIDE_PATH_SVS, HAS_OPENSLIDE_ASSET

IMAGE_1 = IMAGE
if HAS_OPENSLIDE_ASSET:
    IMAGE_2 = SlideReader(SLIDE_PATH_SVS).read_level(-1)[500:1000, 500:1000, :]
else:
    IMAGE_2 = None  # type: ignore[assignment]


def test_macenko_stain_matrix() -> None:
    norm = MachenkoStainNormalizer()
    __, mask = F.get_tissue_mask(IMAGE_1)
    empty_mask = np.zeros_like(IMAGE_1)[..., 0]
    norm.get_stain_matrix(IMAGE_1)
    assert norm.get_stain_matrix(IMAGE_1).shape == (2, 3)
    assert norm.get_stain_matrix(IMAGE_1, mask).shape == (2, 3)
    assert norm.get_stain_matrix(IMAGE_1, empty_mask).shape == (2, 3)


def test_vahadane_stain_matrix() -> None:
    norm = VahadaneStainNormalizer()
    __, mask = F.get_tissue_mask(IMAGE_1)
    empty_mask = np.zeros_like(IMAGE_1)[..., 0]
    norm.get_stain_matrix(IMAGE_1)
    assert norm.get_stain_matrix(IMAGE_1).shape == (2, 3)
    assert norm.get_stain_matrix(IMAGE_1, mask).shape == (2, 3)
    assert norm.get_stain_matrix(IMAGE_1, empty_mask).shape == (2, 3)


def test_macenko_normalizer_fit() -> None:
    norm = MachenkoStainNormalizer()
    if IMAGE_2 is None:
        return pytest.skip("OpenSlide test data or dependency missing")
    norm.fit(IMAGE_2)
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_macenko_normalizer_fit_with_mask() -> None:
    norm = MachenkoStainNormalizer()
    if IMAGE_2 is None:
        return pytest.skip("OpenSlide test data or dependency missing")
    norm.fit(IMAGE_2, tissue_mask=F.get_tissue_mask(IMAGE_2)[1])
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_macenko_normalizer_no_fit() -> None:
    norm = MachenkoStainNormalizer()
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_vahadane_normalizer_fit() -> None:
    norm = VahadaneStainNormalizer()
    if IMAGE_2 is None:
        return pytest.skip("OpenSlide test data or dependency missing")
    norm.fit(IMAGE_2)
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_vahadane_normalizer_fit_with_mask() -> None:
    norm = VahadaneStainNormalizer()
    if IMAGE_2 is None:
        return pytest.skip("OpenSlide test data or dependency missing")
    norm.fit(IMAGE_2, tissue_mask=F.get_tissue_mask(IMAGE_2)[1])
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_vahadane_normalizer_no_fit() -> None:
    norm = VahadaneStainNormalizer()
    assert norm.normalize(IMAGE_1).shape == IMAGE_1.shape


def test_split_stains_macenko() -> None:
    norm = MachenkoStainNormalizer()
    haema, eosin = norm.separate_stains(IMAGE_1)
    assert haema.shape == IMAGE_1.shape
    assert eosin.shape == IMAGE_1.shape


def test_split_stains_vahadane() -> None:
    norm = VahadaneStainNormalizer()
    haema, eosin = norm.separate_stains(IMAGE_1)
    assert haema.shape == IMAGE_1.shape
    assert eosin.shape == IMAGE_1.shape


def test_adjust_stains_macenko() -> None:
    norm = MachenkoStainNormalizer()
    adjusted = norm.adjust_stain_concentrations(
        IMAGE_1,
        haematoxylin_magnitude=0.5,
        eosin_magnitude=1.2,
    )
    assert adjusted.shape == IMAGE_1.shape


def test_adjust_stains_vahadane() -> None:
    norm = VahadaneStainNormalizer()
    adjusted = norm.adjust_stain_concentrations(
        IMAGE_1,
        haematoxylin_magnitude=0.5,
        eosin_magnitude=1.2,
    )
    assert adjusted.shape == IMAGE_1.shape


def test_get_concentrations_macenko() -> None:
    norm = MachenkoStainNormalizer()
    concentrations = norm.get_stain_concentrations(IMAGE_1)
    assert concentrations.shape == (len(IMAGE_1[..., 0].flatten()), 2)


def test_get_concentrations_vahadane() -> None:
    norm = VahadaneStainNormalizer()
    concentrations = norm.get_stain_concentrations(IMAGE_1)
    assert concentrations.shape == (len(IMAGE_1[..., 0].flatten()), 2)
