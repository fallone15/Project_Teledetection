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
from harmonic_regression import fit_harmonic_model, pred_harmonic_model

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

    # ===== Étape 3 : Régression harmonique sur les données 2016 =====

    # Masques : True = pixel invalide (NaN dans le NDVI)
    masks_array = np.isnan(ndvi_array)

    # Fit du modèle sur tous les pixels de l'image
    print("Fitting du modèle harmonique...")
    harmonic_model = fit_harmonic_model(days, ndvi_array, masks_array)
    print("Fit terminé !")

    # Prédictions sur les mêmes dates (pour visualiser l'ajustement)
    predictions = pred_harmonic_model(days, harmonic_model)
    ts_hat = predictions['ts_hat']
    pred_interval = predictions['prediction_interval']

    # === Plot 1 : NDVI observé vs modèle pour le pixel central ===
    fig, ax = plt.subplots(figsize=(12, 5))
    r, c = px_coords

    ndvi_px = ndvi_array[:, r, c]
    hat_px = ts_hat[:, r, c]
    pi_px = pred_interval[:, r, c]

    ax.scatter(days, ndvi_px, color="#8fc0a9", label="NDVI observé", zorder=3)
    ax.plot(days, hat_px, color="#c83349", linewidth=2, label="Modèle harmonique")
    ax.fill_between(days, hat_px - pi_px, hat_px + pi_px,
                     color="#c83349", alpha=0.15, label="Intervalle de prédiction 95%")
    ax.set_ylabel("NDVI", fontsize=15)
    ax.set_ylim([-0.2, 1.1])
    ax.set_title(f"Régression harmonique — pixel ({r}, {c})")
    xticks = np.arange(len(dates))[::6]
    ax.set_xticks(days[xticks], labels=dates[xticks], rotation=45)
    ax.xaxis.grid(True, which="major", linestyle="dashed")
    ax.legend()
    fig.tight_layout()

    # === Diagnostic : stats du modèle ===
    coeff = harmonic_model['coeff']
    amplitude = np.sqrt(coeff[1]**2 + coeff[2]**2)
    std = harmonic_model['std_hat']
    print(f"a0 (moyenne NDVI)  : min={np.nanmin(coeff[0]):.3f}  max={np.nanmax(coeff[0]):.3f}  median={np.nanmedian(coeff[0]):.3f}")
    print(f"Amplitude          : min={np.nanmin(amplitude):.3f}  max={np.nanmax(amplitude):.3f}  median={np.nanmedian(amplitude):.3f}")
    print(f"σ̂ (résidus)        : min={np.nanmin(std):.3f}  max={np.nanmax(std):.3f}  median={np.nanmedian(std):.3f}")
    print(f"Degrés de liberté  : min={np.nanmin(harmonic_model['df']):.0f}  max={np.nanmax(harmonic_model['df']):.0f}")

    # === Plot 2 : Carte des coefficients du modèle ===
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axs[0].imshow(coeff[0], cmap='YlGn', vmin=0, vmax=1)
    axs[0].set_title("a₀ — Moyenne NDVI annuelle")
    plt.colorbar(im0, ax=axs[0], shrink=0.8)

    im1 = axs[1].imshow(amplitude, cmap='OrRd', vmin=0, vmax=np.nanpercentile(amplitude, 98))
    axs[1].set_title("Amplitude du cycle annuel")
    plt.colorbar(im1, ax=axs[1], shrink=0.8)

    im2 = axs[2].imshow(std, cmap='Reds', vmin=0, vmax=np.nanpercentile(std, 98))
    axs[2].set_title("σ̂ — Écart-type résiduel")
    plt.colorbar(im2, ax=axs[2], shrink=0.8)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Paramètres du modèle harmonique", fontsize=14)
    fig.tight_layout()

    plt.show()