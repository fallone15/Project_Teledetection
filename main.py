import os
import yaml
import json

import rasterio
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from download_from_cdse import query_id
from spectral_index import apply_ndvi, apply_crswir
from preprocess import (
    digital_numbers_to_reflectance,
    apply_cloud_mask,
    apply_rgb_compo,
)
from utils import dates_to_days
from harmonic_regression import fit_harmonic_model, pred_harmonic_model
from anomaly_detection import detect_dieback

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
        train_ts[date]['CRSWIR'] = apply_crswir(
            train_ts[date], preprocess=digital_numbers_to_reflectance
        )

        # Étape 2 : appliquer le masque au NDVI, au CRSWIR et à l'image RGB
        mask = apply_cloud_mask(
            train_ts[date], preprocess=digital_numbers_to_reflectance
        )
        train_ts[date]['NDVI'][mask] = np.nan
        train_ts[date]['CRSWIR'][mask] = np.nan
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

    # Make a raster of size (n_days, height, width) of CRSWIR
    crswir_array = np.array([train_ts[date]['CRSWIR'] for date in train_ts])
    height, width = crswir_array.shape[1], crswir_array.shape[2]
    px_coords = height // 2, width // 2

    # Plot the CRSWIR time series
    fig, ax = plt.subplots()
    fontsize = 15
    crswir_px = crswir_array[:, px_coords[0], px_coords[1]]
    ax.scatter(days, crswir_px, color="#e67e22")
    ax.set_ylabel("CRSWIR", fontsize=fontsize)
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

    # ===== Étape 3 : Régression harmonique sur le CRSWIR 2016 =====

    # Masques : True = pixel invalide (NaN dans le CRSWIR)
    masks_array = np.isnan(crswir_array)

    # Fit du modèle sur tous les pixels de l'image
    print("Fitting du modèle harmonique sur CRSWIR...")
    harmonic_model = fit_harmonic_model(days, crswir_array, masks_array)
    print("Fit terminé !")

    # Prédictions sur les mêmes dates (pour visualiser l'ajustement)
    predictions = pred_harmonic_model(days, harmonic_model)
    ts_hat = predictions['ts_hat']
    pred_interval = predictions['prediction_interval']

    # === Plot 1 : CRSWIR observé vs modèle pour le pixel central ===
    fig, ax = plt.subplots(figsize=(12, 5))
    r, c = px_coords

    crswir_px = crswir_array[:, r, c]
    hat_px = ts_hat[:, r, c]
    pi_px = pred_interval[:, r, c]

    ax.scatter(days, crswir_px, color="#e67e22", label="CRSWIR observé", zorder=3)
    ax.plot(days, hat_px, color="#c83349", linewidth=2, label="Modèle harmonique")
    ax.fill_between(days, hat_px - pi_px, hat_px + pi_px,
                     color="#c83349", alpha=0.15, label="Intervalle de prédiction 95%")
    ax.set_ylabel("CRSWIR", fontsize=15)
    ax.set_title(f"Régression harmonique CRSWIR — pixel ({r}, {c})")
    xticks = np.arange(len(dates))[::6]
    ax.set_xticks(days[xticks], labels=dates[xticks], rotation=45)
    ax.xaxis.grid(True, which="major", linestyle="dashed")
    ax.legend()
    fig.tight_layout()

    # === Diagnostic : stats du modèle ===
    coeff = harmonic_model['coeff']
    amplitude = np.sqrt(coeff[1]**2 + coeff[2]**2)
    std = harmonic_model['std_hat']
    print(f"a0 (moyenne CRSWIR): min={np.nanmin(coeff[0]):.3f}  max={np.nanmax(coeff[0]):.3f}  median={np.nanmedian(coeff[0]):.3f}")
    print(f"Amplitude          : min={np.nanmin(amplitude):.3f}  max={np.nanmax(amplitude):.3f}  median={np.nanmedian(amplitude):.3f}")
    print(f"σ̂ (résidus)        : min={np.nanmin(std):.3f}  max={np.nanmax(std):.3f}  median={np.nanmedian(std):.3f}")
    print(f"Degrés de liberté  : min={np.nanmin(harmonic_model['df']):.0f}  max={np.nanmax(harmonic_model['df']):.0f}")

    # === Plot 2 : Carte des coefficients du modèle ===
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axs[0].imshow(coeff[0], cmap='YlOrBr', vmin=np.nanpercentile(coeff[0], 2), vmax=np.nanpercentile(coeff[0], 98))
    axs[0].set_title("a₀ — Moyenne CRSWIR annuelle")
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
    fig.suptitle("Paramètres du modèle harmonique (CRSWIR)", fontsize=14)
    fig.tight_layout()

    plt.show()

    # ===== Étape 5 : Détection du dépérissement forestier sur 2018 =====
    if len(years) > 1:
        print("\n===== Étape 5 : Détection du dépérissement sur les données 2018 =====")
        test_data_folder = os.path.join(data_folder, years[1])

        with open(os.path.join(test_data_folder, bands[0], 'job-results.json'), 'r') as f:
            job_results_2018 = json.load(f)

        img_files_2018 = list(job_results_2018['assets'].keys())
        img_dates_2018 = sorted([file.split('_')[-1].split('Z')[0] for file in img_files_2018])

        print(f"Chargement de {len(img_dates_2018)} images Sentinel-2 pour l'année 2018...")
        test_ts = {}
        for date in img_dates_2018:
            test_ts[date] = dict(
                (
                    band,
                    rasterio.open(
                        os.path.join(
                            test_data_folder,
                            band,
                            f'SENTINEL2_L2A_{band}_{date}Z.tif',
                        )
                    ),
                )
                for band in bands
            )

            # Calcul du CRSWIR 2018
            test_ts[date]['CRSWIR'] = apply_crswir(
                test_ts[date], preprocess=digital_numbers_to_reflectance
            )

            # Masquage nuages / non-végétation
            mask = apply_cloud_mask(
                test_ts[date], preprocess=digital_numbers_to_reflectance
            )
            test_ts[date]['CRSWIR'][mask] = np.nan
            test_ts[date]['mask'] = mask

            for band in bands:
                test_ts[date][band].close()

        dates_2018 = np.array(list(test_ts.keys()))
        days_2018 = dates_to_days(
            np.array([datetime.strptime(date, '%Y-%m-%d') for date in test_ts])
        )

        crswir_2018 = np.array([test_ts[date]['CRSWIR'] for date in test_ts])
        masks_2018 = np.isnan(crswir_2018)

        # 1. Prédire le comportement normal attendu en 2018 avec le modèle harmonique 2016
        print("Prédiction des valeurs de référence pour 2018 avec le modèle harmonique...")
        pred_2018 = pred_harmonic_model(days_2018, harmonic_model)
        ts_hat_2018 = pred_2018['ts_hat']
        pi_2018 = pred_2018['prediction_interval']

        # 2. Détection des anomalies de dépérissement (CRSWIR_obs > seuil haut à 95%)
        print("Détection des anomalies de dépérissement (detect_dieback)...")
        dieback_results = detect_dieback(crswir_2018, ts_hat_2018, pi_2018, masks_2018)

        n_pixels_total = dieback_results['dieback_map'].size
        n_pixels_dieback = np.sum(dieback_results['dieback_map'])
        pct_dieback = (n_pixels_dieback / n_pixels_total) * 100
        print(f"Résultats 2018 : {n_pixels_dieback} pixels en dépérissement sur {n_pixels_total} ({pct_dieback:.2f}%)")

        # Plot 1 : Cartes spatiales de dépérissement
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        im0 = axs[0].imshow(dieback_results['dieback_map'], cmap='Reds')
        axs[0].set_title(f"Carte binaire de dépérissement 2018 ({pct_dieback:.1f}% de la zone)")
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        im1 = axs[1].imshow(dieback_results['anomaly_count'], cmap='YlOrRd')
        axs[1].set_title("Nombre de dates en anomalie par pixel (2018)")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        plt.colorbar(im1, ax=axs[1], shrink=0.8)

        fig.suptitle("Étape 5 — Détection du dépérissement forestier (Tree Dieback en 2018)", fontsize=14)
        fig.tight_layout()
        plt.show()

        # Plot 2 : Série temporelle continue complète (2016 -> 2018)
        # Montre la transition entre la période saine de référence et le décrochage en 2018
        r_d, c_d = np.unravel_index(np.argmax(dieback_results['anomaly_count']), dieback_results['anomaly_count'].shape)

        # Concaténer 2016-2017 et 2018
        all_days = np.concatenate([days, days_2018])
        all_dates = np.concatenate([dates, dates_2018])
        all_pred_px = np.concatenate([ts_hat[:, r_d, c_d], ts_hat_2018[:, r_d, c_d]])
        all_pi_px = np.concatenate([pred_interval[:, r_d, c_d], pi_2018[:, r_d, c_d]])

        sort_idx = np.argsort(all_days)
        all_days = all_days[sort_idx]
        all_dates = all_dates[sort_idx]
        all_pred_px = all_pred_px[sort_idx]
        all_pi_px = all_pi_px[sort_idx]

        fig, ax = plt.subplots(figsize=(15, 6))

        # 1. Données 2016-2017 (période de référence saine en vert)
        ax.scatter(days, crswir_array[:, r_d, c_d], color="#27ae60", label="2016–2017 : Observations saines (référence)", zorder=3, alpha=0.85)

        # 2. Données 2018 : normales (orange) vs anomalies (triangles rouges)
        obs_18 = crswir_2018[:, r_d, c_d]
        upper_18 = ts_hat_2018[:, r_d, c_d] + pi_2018[:, r_d, c_d]
        is_anomaly_18 = (obs_18 > upper_18) & ~masks_2018[:, r_d, c_d]
        is_normal_18 = ~is_anomaly_18 & ~masks_2018[:, r_d, c_d]

        ax.scatter(days_2018[is_normal_18], obs_18[is_normal_18], color="#e67e22", label="2018 : Observations dans la norme", zorder=3)
        ax.scatter(days_2018[is_anomaly_18], obs_18[is_anomaly_18], color="#c0392b", marker="^", s=80, label="2018 : Anomalies (Dépérissement détecté)", zorder=4)

        # 3. Modèle harmonique continu et son enveloppe à 95%
        ax.plot(all_days, all_pred_px, color="#2c3e50", linewidth=2, label="Modèle harmonique (calibré sur 2016)")
        ax.fill_between(all_days, all_pred_px - all_pi_px, all_pred_px + all_pi_px,
                        color="#3498db", alpha=0.15, label="Intervalle normal attendu à 95%")
        ax.plot(all_days, all_pred_px + all_pi_px, color="#e74c3c", linestyle="--", linewidth=1.5, label="Seuil haut d'alerte")

        # Séparation visuelle entre 2016-2017 et 2018
        split_day = days_2018[0]
        ax.axvline(split_day, color="black", linestyle=":", linewidth=2, label="Début 2018")

        ax.set_ylabel("CRSWIR", fontsize=14)
        ax.set_title(f"Suivi temporel continu (2016 → 2018) — Pixel ({r_d}, {c_d}) en dépérissement", fontsize=14, fontweight="bold")
        xticks = np.arange(len(all_dates))[::8]
        ax.set_xticks(all_days[xticks], labels=all_dates[xticks], rotation=45)
        ax.xaxis.grid(True, which="major", linestyle="dashed", alpha=0.5)
        ax.legend(loc="upper left")
        fig.tight_layout()
        plt.show()