import os
import yaml
import json

import rasterio
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from download_from_cdse import query_id
from spectral_index import apply_ndvi
from preprocess import (
    digital_numbers_to_reflectance,
    apply_cloud_mask,
    apply_rgb_compo,
)
from utils import dates_to_days

if __name__ == "__main__":
    with open('cdse_query.yaml', 'r', encoding='utf-8') as file:
        query = yaml.safe_load(file)

    bands = query['bands']
    data_folder = os.path.join(
        query['data_folder'], query_id(query, temporal_id=False)
    )
    years = sorted(os.listdir(data_folder))

    train_data_folder = os.path.join(data_folder, years[0])

    # Load Sentinel-2 data
    with open(os.path.join(train_data_folder, bands[0], 'job-results.json'), 'r') as f:
        job_results = json.load(f)

    img_files = list(job_results['assets'].keys())
    img_dates = [file.split('_')[-1].split('Z')[0] for file in img_files]


    reflectance_ts = {}
    # Make Sentinel-2 time series
    train_ts = {}
    for date in img_dates:
        train_ts[date] = dict(
            (
                band,
                rasterio.open(
                    os.path.join(
                        train_data_folder,
                        band,
                        f'SENTINEL2_L2A_{band}_{date}Z.tif',
                    )
                ),
            )
            for band in bands
        )

        train_ts[date]['NDVI'] = apply_ndvi(
            train_ts[date], preprocess=digital_numbers_to_reflectance
        )

        # Étape 2 : appliquer le masque au NDVI et à l'image RGB
        mask = apply_cloud_mask(
            train_ts[date], preprocess=digital_numbers_to_reflectance
        )
        train_ts[date]['NDVI'][mask] = np.nan
        train_ts[date]['RGB'] = apply_rgb_compo(
            train_ts[date],
            preprocess=digital_numbers_to_reflectance,
            mask=mask,
        )
    
        for band in bands:
            train_ts[date][band].close()
    
    dates = np.array(list(train_ts.keys()))
    n_dates = len(dates)
    days = dates_to_days(
        np.array([datetime.strptime(date, '%Y-%m-%d') for date in train_ts])
    )
    
    """--------------------------------------------------------------------------------------
    """
    # Stocker les réflectances de chaque bande
    reflectance_ts[date] = {}

    for band in bands:
        raw = train_ts[date][band].read(1, masked=True)

        reflectance = digital_numbers_to_reflectance(
            raw.astype(np.float32)
        )

        reflectance_ts[date][band] = np.ma.asarray(
            reflectance
        ).filled(np.nan)



    """--------------------------------------------------------------------------------------"""

    # Make a raster of size (n_days, height, width) of NDVI
    ndvi_array = np.array([train_ts[date]['NDVI'] for date in train_ts])
    height, width = ndvi_array.shape[1], ndvi_array.shape[2]
    px_coords = height // 2, width // 2

    # Plot the NDVI time series
    fig, ax = plt.subplots()
    fontsize = 15
    ndvi_px = ndvi_array[:, px_coords[0], px_coords[1]]
    ax.scatter(days, ndvi_px, color="#8fc0a9")
    ax.set_ylabel("NDVI", fontsize=fontsize)
    ax.set_ylim([-1, 1])
    xticks = np.arange(len(dates))[::6]
    xticklabels = dates[xticks]
    ax.set_xticks(days[xticks], labels=xticklabels, rotation=45)
    ax.xaxis.grid(True, which="major", linestyle="dashed")
    fig.tight_layout()

    # Plot RGB composition over the time series
    n_rows, n_cols = 2, 4
    images_per_page = n_rows * n_cols

    for start in range(0, n_dates, images_per_page):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 8))
        page_dates = dates[start:start + images_per_page]

        for ax, date in zip(axs.flat, page_dates):
            ax.imshow(train_ts[date]['RGB'])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(date)

        for ax in axs.flat[len(page_dates):]:
            ax.set_visible(False)

        fig.tight_layout()

    plt.show()

    # Histogrammes de réflectance pour comparer les valeurs seuils
    # Afficher les histogrammes de réflectance

    date = list(reflectance_ts.keys())[0]

    band = "B02"

    reflectance = reflectance_ts[date][band]

    # Enlever les valeurs invalides
    values = reflectance[np.isfinite(reflectance)]

    plt.figure(figsize=(8, 5))

    plt.hist(values, bins=50, color="skyblue", edgecolor="black")

    plt.xlabel("Réflectance")
    plt.ylabel("Nombre de pixels")
    plt.title(f"Histogramme de réflectance {band} - {date}")
    
    plt.show()