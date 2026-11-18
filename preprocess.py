
from typing import Callable, Dict

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

from spectral_index import apply_ndvi
from utils import rgb_render


def digital_numbers_to_reflectance(raster: np.ndarray):
    """Convert a Sentinel-2 raster provided in digital numbers to reflectance
    for data acquired from 2016 to 2020. In this period, the formula for every
    S2 bands is : REFLECTANCE = DIGITAL_NUMBERS / 10000.
    """
    raster = raster / 10000
    raster = np.clip(raster, a_min=0, a_max=None)
    return raster


def apply_cloud_mask(
    rasters: Dict[str, rasterio.io.DatasetReader],
    preprocess: Callable = digital_numbers_to_reflectance,
):
    """Étape 2 : exclure nuages, ombres probables et non-végétation.

    True = pixel à ignorer. Le nom initial du professeur est conservé.
    Les seuils sont pédagogiques, à vérifier visuellement sur les données.
    Pour des entrées déjà en réflectance : preprocess=lambda x: x.
    """
    required = ("B02", "B03", "B04", "B08")
    missing = [band for band in required if band not in rasters]
    if missing:
        raise ValueError("Bandes manquantes : " + ", ".join(missing))

    reference = rasters["B02"]
    invalid = np.zeros(reference.shape, dtype=bool)
    bands = {}

    for band in required:
        dataset = rasters[band]
        if (dataset.shape != reference.shape
                or dataset.transform != reference.transform
                or dataset.crs != reference.crs):
            raise ValueError("Les bandes doivent avoir la même grille.")

        raw = dataset.read(1, masked=True)
        invalid |= np.ma.getmaskarray(raw) | (raw.data == 0)
        data = np.ma.asarray(preprocess(raw), dtype=np.float32)
        values = data.filled(np.nan)
        invalid |= np.ma.getmaskarray(data) | ~np.isfinite(values)
        bands[band] = values

    # La fonction NDVI existe déjà dans spectral_index.py.
    # Les flottants évitent un débordement lors de la soustraction de uint16.
    ndvi = np.ma.asarray(apply_ndvi(
        rasters, preprocess=lambda x: np.asanyarray(preprocess(x), dtype=float)
    ), dtype=float).filled(np.nan)
    invalid |= ~np.isfinite(ndvi)

    brightness = (bands["B02"] + bands["B03"] + bands["B04"]) / 3
    clouds = (
        (bands["B02"] > 0.2)
        & (bands["B03"] > 0.2)
        & (bands["B04"] > 0.2)
        & (brightness > 0.25)
        & (ndvi < 0.4)
        & ~invalid
    )
    non_vegetation = (ndvi < 0.3) & ~invalid
    return invalid | clouds | non_vegetation

def apply_rgb_compo(
    rasters: Dict[str, rasterio.io.DatasetReader],
    preprocess: Callable = lambda x: x,
    mask: np.ndarray = None,
    render: bool = True,
):
    """Image RGB (B04, B03, B02), forme (height, width, 3).

    render=True : image uint8 pour affichage, pixels exclus en noir.
    render=False : valeurs scientifiques flottantes, pixels exclus en NaN.
    Le masque n'est appliqué que s'il est transmis explicitement par main.py.
    """
    rgb_bands = ('B04', 'B03', 'B02')
    missing = [band for band in rgb_bands if band not in rasters]
    if missing:
        raise ValueError('Bandes manquantes : ' + ', '.join(missing))
    reference = rasters['B04']
    invalid = np.zeros(reference.shape, dtype=bool)
    bands = {}
    for band in rgb_bands:
        dataset = rasters[band]
        if (dataset.shape != reference.shape
                or dataset.transform != reference.transform
                or dataset.crs != reference.crs):
            raise ValueError('Les bandes RGB doivent avoir la même grille.')
        raw = dataset.read(1, masked=True)
        invalid |= np.ma.getmaskarray(raw) | (raw.data == 0)
        data = np.ma.asarray(preprocess(raw), dtype=np.float32)
        values = data.filled(np.nan)
        invalid |= np.ma.getmaskarray(data) | ~np.isfinite(values)
        bands[band] = values
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != invalid.shape:
            raise ValueError('Le masque doit avoir les dimensions des bandes RGB.')
        invalid |= mask
    composition = np.stack([bands[b] for b in ('B04', 'B03', 'B02')])
    composition[:, invalid] = np.nan
    if not render:
        return np.moveaxis(composition, 0, -1)
    if invalid.all():
        return np.zeros((*invalid.shape, 3), dtype=np.uint8)

    # Éviter une division par zéro si un canal est constant.
    dmin = np.nanpercentile(composition, 2, axis=(1, 2))
    dmax = np.nanpercentile(composition, 98, axis=(1, 2))
    dmax = np.maximum(dmax, dmin + np.maximum(np.abs(dmin) * 1e-6, 1e-6))
    image, _, _ = rgb_render(composition, bands=[0, 1, 2], mask=invalid,
                              dmin=dmin, dmax=dmax)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image[invalid] = 0
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)
