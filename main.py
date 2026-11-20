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
from anomaly_detection import (
    detect_dieback,
    distinguish_clearcut_vs_dieback,
    distinguish_drought_vs_bark_beetle,
    classify_forest_disturbances,
)
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

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

    output_fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(output_fig_dir, exist_ok=True)
    print(f"Dossier de sauvegarde des figures : {output_fig_dir}")

    # Plot the CRSWIR time series
    fig, ax = plt.subplots(figsize=(10, 5))
    fontsize = 15
    crswir_px = crswir_array[:, px_coords[0], px_coords[1]]
    ax.scatter(days, crswir_px, color="#e67e22")
    ax.set_ylabel("CRSWIR", fontsize=fontsize)
    xticks = np.arange(len(dates))[::6]
    xticklabels = dates[xticks]
    ax.set_xticks(days[xticks], labels=xticklabels, rotation=45)
    ax.xaxis.grid(True, which="major", linestyle="dashed")
    fig.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "01_crswir_scatter_2016.png"), dpi=200, bbox_inches="tight")
    print("Enregistré : 01_crswir_scatter_2016.png")

    # Plot RGB composition over the time series
    n_rows, n_cols = 2, 4
    images_per_page = n_rows * n_cols

    page_num = 1
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
        fig.savefig(os.path.join(output_fig_dir, f"02_rgb_timeseries_page_{page_num}.png"), dpi=150, bbox_inches="tight")
        print(f"Enregistré : 02_rgb_timeseries_page_{page_num}.png")
        page_num += 1

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
    fig.savefig(os.path.join(output_fig_dir, "04_harmonic_regression_pixel_2016.png"), dpi=200, bbox_inches="tight")
    print("Enregistré : 03_harmonic_regression_pixel_2016.png")

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
    fig.savefig(os.path.join(output_fig_dir, "04_harmonic_parameters_maps_crswir.png"), dpi=200, bbox_inches="tight")
    print("Enregistré : 04_harmonic_parameters_maps_crswir.png")

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

            # Calcul du CRSWIR et du NDVI 2018
            test_ts[date]['CRSWIR'] = apply_crswir(
                test_ts[date], preprocess=digital_numbers_to_reflectance
            )
            test_ts[date]['NDVI'] = apply_ndvi(
                test_ts[date], preprocess=digital_numbers_to_reflectance
            )

            # Masquage nuages / non-végétation
            mask = apply_cloud_mask(
                test_ts[date], preprocess=digital_numbers_to_reflectance
            )
            test_ts[date]['CRSWIR'][mask] = np.nan
            test_ts[date]['NDVI'][mask] = np.nan
            test_ts[date]['mask'] = mask

            for band in bands:
                test_ts[date][band].close()

        dates_2018 = np.array(list(test_ts.keys()))
        days_2018 = dates_to_days(
            np.array([datetime.strptime(date, '%Y-%m-%d') for date in test_ts])
        )

        crswir_2018 = np.array([test_ts[date]['CRSWIR'] for date in test_ts])
        ndvi_2018 = np.array([test_ts[date]['NDVI'] for date in test_ts])
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
        fig.savefig(os.path.join(output_fig_dir, "05_dieback_maps_2018.png"), dpi=200, bbox_inches="tight")
        print("Enregistré : 05_dieback_maps_2018.png")
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
        fig.savefig(os.path.join(output_fig_dir, "06_dieback_timeseries_2016_2018.png"), dpi=200, bbox_inches="tight")
        print("Enregistré : 06_dieback_timeseries_2016_2018.png")
        plt.show()

        # ===== Étape 6 : Discrimination Coupes Rases vs Scolytes =====
        print("\n===== Étape 6 : Discrimination Coupes Rases vs Scolytes =====")
        clearcut_results = distinguish_clearcut_vs_dieback(
            ndvi_2018,
            dieback_results,
            ndvi_clearcut_threshold=0.30,
        )
        n_clearcut = np.sum(clearcut_results['clearcut_map'])
        n_standing = np.sum(clearcut_results['standing_dieback_map'])
        pct_clearcut = (n_clearcut / n_pixels_total) * 100
        pct_standing = (n_standing / n_pixels_total) * 100
        print(f"Coupes rases détectées (sol nu, NDVI < 0.30)        : {n_clearcut} pixels ({pct_clearcut:.2f}%)")
        print(f"Dépérissement sur pied (arbres morts, NDVI >= 0.30)  : {n_standing} pixels ({pct_standing:.2f}%)")

        # Plot Étape 6
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].imshow(clearcut_results['clearcut_map'], cmap='YlOrBr')
        axs[0].set_title(f"Coupes rases — Clearcuts ({pct_clearcut:.1f}%)")
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].imshow(clearcut_results['standing_dieback_map'], cmap='Reds')
        axs[1].set_title(f"Dépérissement sur pied ({pct_standing:.1f}%)")
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        im_ndvi = axs[2].imshow(clearcut_results['post_ndvi_min'], cmap='RdYlGn', vmin=0.1, vmax=0.8)
        axs[2].set_title("NDVI post-anomalie minimal")
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        plt.colorbar(im_ndvi, ax=axs[2], shrink=0.8)

        fig.suptitle("Étape 6 — Discrimination : Coupes rases vs Dépérissement sur pied", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_fig_dir, "07_step6_clearcut_vs_dieback.png"), dpi=200, bbox_inches="tight")
        print("Enregistré : 07_step6_clearcut_vs_dieback.png")
        plt.show()

        # ===== Étape 7 : Discrimination Sécheresse vs Scolytes =====
        print("\n===== Étape 7 : Discrimination Sécheresse vs Scolytes =====")
        amplitude_2016 = np.sqrt(coeff[1]**2 + coeff[2]**2)
        drought_beetle_results = distinguish_drought_vs_bark_beetle(
            crswir_2018,
            ts_hat_2018,
            pi_2018,
            masks_2018,
            dates_2018,
            standing_dieback_map=clearcut_results['standing_dieback_map'],
            amplitude_2016=amplitude_2016,
            autumn_start_date='2018-09-15',
        )

        n_drought = np.sum(drought_beetle_results['drought_map'])
        n_beetle = np.sum(drought_beetle_results['bark_beetle_map'])
        n_deciduous = np.sum(drought_beetle_results['deciduous_dieback_map'])
        pct_drought = (n_drought / n_pixels_total) * 100
        pct_beetle = (n_beetle / n_pixels_total) * 100
        pct_deciduous = (n_deciduous / n_pixels_total) * 100

        print(f"Stress sécheresse réversible (récupération automne) : {n_drought} pixels ({pct_drought:.2f}%)")
        print(f"Attaques de scolytes confirmées (résineux, irrév.)  : {n_beetle} pixels ({pct_beetle:.2f}%)")
        print(f"Dépérissement feuillus / autres                     : {n_deciduous} pixels ({pct_deciduous:.2f}%)")

        # Plot Étape 7
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].imshow(drought_beetle_results['drought_map'], cmap='YlGn')
        axs[0].set_title(f"Sécheresse réversible ({pct_drought:.1f}%)")
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].imshow(drought_beetle_results['is_conifer'], cmap='Greens')
        axs[1].set_title("Masque des résineux (faible amplitude 2016)")
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        axs[2].imshow(drought_beetle_results['bark_beetle_map'], cmap='Reds')
        axs[2].set_title(f"Attaques de scolytes confirmées ({pct_beetle:.1f}%)")
        axs[2].set_xticks([])
        axs[2].set_yticks([])

        fig.suptitle("Étape 7 — Discrimination : Sécheresse vs Scolytes (Ips typographus)", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_fig_dir, "09_step7_drought_vs_bark_beetle.png"), dpi=200, bbox_inches="tight")
        print("Enregistré : 08_step7_drought_vs_bark_beetle.png")
        plt.show()

        # ===== Synthèse finale multi-classes (Étapes 5, 6, 7) =====
        print("\n===== Synthèse globale : Cartographie complète des perturbations =====")
        final_classification = np.zeros(dieback_results['dieback_map'].shape, dtype=int)
        final_classification[clearcut_results['clearcut_map']] = 1
        final_classification[drought_beetle_results['drought_map']] = 2
        final_classification[drought_beetle_results['bark_beetle_map']] = 3
        final_classification[drought_beetle_results['deciduous_dieback_map']] = 4

        cmap_classes = ListedColormap(['#ecf0f1', '#795548', '#f39c12', '#c0392b', '#8e44ad'])
        norm_classes = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap_classes.N)

        fig, ax = plt.subplots(figsize=(10, 8))
        im_final = ax.imshow(final_classification, cmap=cmap_classes, norm=norm_classes)
        ax.set_title("Cartographie finale des perturbations forestières 2018 (Vosges / Épinal)\nChaîne complète FORDEAD (Étapes 5, 6 & 7)", fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        legend_labels = [
            f"Forêt saine / Non perturbée ({100 - pct_dieback:.1f}%)",
            f"Coupes rases — Clearcut ({pct_clearcut:.1f}%)",
            f"Stress sécheresse réversible ({pct_drought:.1f}%)",
            f"Scolytes sur conifères — Ips typographus ({pct_beetle:.1f}%)",
            f"Dépérissement feuillus / autre ({pct_deciduous:.1f}%)"
        ]
        patches = [
            mpatches.Patch(color=cmap_classes.colors[i], label=legend_labels[i])
            for i in range(5)
        ]
        ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=11)

        fig.tight_layout()
        fig.savefig(os.path.join(output_fig_dir, "09_final_forest_classification.png"), dpi=200, bbox_inches="tight")
        print("Enregistré : 09_final_forest_classification.png")
        plt.show()