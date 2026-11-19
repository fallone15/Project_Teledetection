from typing import Dict, Callable


def apply_ndvi(raster_dst: Dict, preprocess: Callable = lambda x: x, eps: float = 1e-4):
    """Normalized Difference Vegetation Index.
    Formula for Sentinel-2 bands: (B08-B04)/(B08+B04)

    Args:
        rasters: dictionary with keys B01, ..., B12 mapping to rasterio.io.DatasetReader datasets

    Returns:
        an array of NDVI values.
    """
    B08 = preprocess(raster_dst['B08'].read(1))
    B04 = preprocess(raster_dst['B04'].read(1))
    return (B08-B04)/(B08+B04+eps)


def apply_swir(raster_dst: Dict, preprocess: Callable = lambda x: x, eps: float = 1e-4):
    """Indice SWIR pour Sentinel-2.
    Formule : (B08 - B11) / (B08 + B11)

    Args:
        raster_dst: dictionnaire avec les clés B01, ..., B12 mappant vers rasterio

    Returns:
        un tableau numpy des valeurs de l'indice SWIR.
    """
    B08 = preprocess(raster_dst['B08'].read(1))
    B11 = preprocess(raster_dst['B11'].read(1))
    return (B08 - B11) / (B08 + B11 + eps)


def apply_crswir(raster_dst: Dict, preprocess: Callable = lambda x: x, eps: float = 1e-4):
    """Continuum Removal Short-Wave Infrared (CRSWIR) pour Sentinel-2 (méthode FORDEAD).
    Formule : B11 / (B8A + ((B12 - B8A) / (2185.7 - 864)) * (1610.4 - 864))

    Mesure la profondeur de l'absorption d'eau dans la canopée (sensible aux scolytes / dépérissement).

    Args:
        raster_dst: dictionnaire contenant au moins les clés 'B8A', 'B11', 'B12'

    Returns:
        un tableau numpy des valeurs de l'indice CRSWIR.
    """
    B8A = preprocess(raster_dst['B8A'].read(1))
    B11 = preprocess(raster_dst['B11'].read(1))
    B12 = preprocess(raster_dst['B12'].read(1))

    # Ligne de continuum entre B8A (864 nm) et B12 (2185.7 nm) interpolée à B11 (1610.4 nm)
    continuum = B8A + ((B12 - B8A) / (2185.7 - 864.0)) * (1610.4 - 864.0)
    return B11 / (continuum + eps)

