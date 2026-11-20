import numpy as np
from typing import Dict, Optional


def detect_dieback(
    obs_crswir: np.ndarray,
    pred_crswir: np.ndarray,
    pred_intervals: np.ndarray,
    masks: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Étape 5 : Méthode de détection du dépérissement forestier (tree dieback) sur les données 2018.

    Un pixel présente une anomalie lorsque le CRSWIR observé dépasse
    le seuil haut de l'intervalle de prédiction à 95% :
        CRSWIR_obs(t) > CRSWIR_pred(t) + prediction_interval(t)

    Args:
        obs_crswir: tableau des valeurs CRSWIR observées en 2018, de forme (n_days, H, W)
        pred_crswir: tableau des prédictions du modèle harmonique, même forme
        pred_intervals: demi-largeur de l'intervalle de prédiction à 95%, même forme
        masks: masque booléen des pixels invalides/nuages (True = invalide à ignorer)

    Returns:
        dictionnaire contenant :
            - 'residuals': écarts (obs - pred) de forme (n_days, H, W)
            - 'upper_threshold': seuil haut de prédiction à 95% (n_days, H, W)
            - 'anomaly_mask': booléen indiquant les observations en anomalie (n_days, H, W)
            - 'dieback_map': carte binaire (H, W), True si le pixel a au moins une anomalie
            - 'anomaly_count': nombre de dates en anomalie par pixel (H, W)
            - 'max_anomaly': amplitude maximale du dépassement par rapport au seuil haut (H, W)
    """
    valid = ~masks & np.isfinite(obs_crswir) & np.isfinite(pred_crswir) & np.isfinite(pred_intervals)

    # Résidus (obs - pred)
    residuals = np.where(valid, obs_crswir - pred_crswir, np.nan)

    # Seuil haut de l'intervalle de prédiction à 95%
    upper_threshold = pred_crswir + pred_intervals

    # Dépassement du seuil haut (perte anormale d'eau foliaire dans la canopée)
    anomaly_mask = valid & (obs_crswir > upper_threshold)

    # Nombre de dates où l'anomalie est observée par pixel
    anomaly_count = np.sum(anomaly_mask, axis=0)

    # Carte binaire de dépérissement : le pixel a au moins une anomalie en 2018
    dieback_map = anomaly_count > 2

    # Excès maximal au-dessus du seuil haut
    excess = np.where(anomaly_mask, obs_crswir - upper_threshold, 0.0)
    max_anomaly = np.max(excess, axis=0)

    return {
        'residuals': residuals,
        'upper_threshold': upper_threshold,
        'anomaly_mask': anomaly_mask,
        'dieback_map': dieback_map,
        'anomaly_count': anomaly_count,
        'max_anomaly': max_anomaly,
    }


def distinguish_clearcut_vs_dieback(
    obs_ndvi: np.ndarray,
    dieback_results: Dict[str, np.ndarray],
    ndvi_clearcut_threshold: float = 0.30,
) -> Dict[str, np.ndarray]:
    """
    Étape 6 : Amélioration de la méthode pour distinguer les coupes rases
    (clearcut harvesting) des attaques de scolytes / dépérissement sur pied.

    Principe biophysique :
        - Coupe rase : abattage complet des arbres et mise à nu du sol minéral.
          Le NDVI s'effondre de manière brutale et durable (NDVI < 0.30).
        - Dépérissement sur pied (scolytes) : l'arbre reste debout avec sa structure
          ligneuse et ses aiguilles rousses ou branches défoliées. Le NDVI diminue
          mais reste modéré (NDVI >= 0.30, typiquement entre 0.35 et 0.55).

    Args:
        obs_ndvi: cube NDVI observé en 2018, de forme (n_days, H, W)
        dieback_results: dictionnaire issu de detect_dieback() (Étape 5)
        ndvi_clearcut_threshold: seuil de NDVI sous lequel le pixel est classé en coupe rase (défaut : 0.30)

    Returns:
        dictionnaire contenant :
            - 'clearcut_map': booléen (H, W), True si coupe rase
            - 'standing_dieback_map': booléen (H, W), True si dépérissement sur pied (candidat scolyte / sécheresse)
            - 'post_ndvi_min': float (H, W), valeur minimale de NDVI post-anomalie
            - 'post_ndvi_mean': float (H, W), valeur moyenne de NDVI post-anomalie
            - 'ndvi_drop': float (H, W), chute maximale de NDVI observée en 2018
    """
    anomaly_mask = dieback_results['anomaly_mask']  # (n_days, H, W)
    dieback_map = dieback_results['dieback_map']    # (H, W)

    # Masque temporel : True pour toutes les dates suivant la 1ère anomalie du pixel
    cumulative_anomaly = np.cumsum(anomaly_mask, axis=0)
    post_anomaly_mask = cumulative_anomaly > 0

    # NDVI valide post-anomalie
    valid_post_ndvi = np.where(
        post_anomaly_mask & np.isfinite(obs_ndvi),
        obs_ndvi,
        np.nan
    )

    with np.errstate(all='ignore'):
        post_ndvi_min = np.nanmin(valid_post_ndvi, axis=0)
        post_ndvi_mean = np.nanmean(valid_post_ndvi, axis=0)
        overall_ndvi_min = np.nanmin(obs_ndvi, axis=0)
        overall_ndvi_max = np.nanmax(obs_ndvi, axis=0)

    # Chute globale de NDVI au cours de l'année
    ndvi_drop = overall_ndvi_max - overall_ndvi_min

    # Si un pixel a peu d'observations post-anomalie, on utilise le min global
    effective_min_ndvi = np.where(np.isfinite(post_ndvi_min), post_ndvi_min, overall_ndvi_min)

    # Classification
    # Coupe rase : pixel en dépérissement ayant atteint un niveau de sol nu (NDVI < seuil)
    clearcut_map = dieback_map & np.isfinite(effective_min_ndvi) & (effective_min_ndvi < ndvi_clearcut_threshold)

    # Dépérissement sur pied : pixel en dépérissement conservant un couvert végétal / ligneux résiduel
    standing_dieback_map = dieback_map & (~clearcut_map)

    return {
        'clearcut_map': clearcut_map,
        'standing_dieback_map': standing_dieback_map,
        'post_ndvi_min': post_ndvi_min,
        'post_ndvi_mean': post_ndvi_mean,
        'ndvi_drop': ndvi_drop,
    }


def distinguish_drought_vs_bark_beetle(
    obs_crswir: np.ndarray,
    pred_crswir: np.ndarray,
    pred_intervals: np.ndarray,
    masks: np.ndarray,
    dates: np.ndarray,
    standing_dieback_map: np.ndarray,
    amplitude_2016: np.ndarray,
    autumn_start_date: str = '2018-09-15',
    amplitude_conifer_threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Étape 7 : Amélioration de la méthode pour distinguer le stress sécheresse (drought)
    des attaques de scolytes (bark beetle infestations).

    Deux critères discriminants majeurs :
    1. Réversibilité en Automne :
       - Sécheresse : réversible. Dès les pluies d'automne (septembre-novembre),
         les arbres vivants se réhydratent, et le CRSWIR redescend dans l'intervalle normal.
       - Scolytes : irréversible. L'arbre est mort (système vasculaire détruit),
         le CRSWIR reste anormalement élevé en automne.
    2. Filtre d'Essence Forestière (Amplitude saisonnière 2016) :
       - Les scolytes typographes (Ips typographus) s'attaquent exclusivement aux conifères (épicéas).
       - Les conifères ont un feuillage pérenne et une FAIBLE amplitude saisonnière.
       - Les feuillus (hêtres, chênes) perdent leurs feuilles et ont une FORTE amplitude saisonnière.
       - Les pixels à forte amplitude ayant souffert en été correspondent à un stress de feuillus,
         non à des scolytes.

    Args:
        obs_crswir: tableau CRSWIR observé 2018 (n_days, H, W)
        pred_crswir: prédictions harmoniques 2018 (n_days, H, W)
        pred_intervals: intervalles de prédiction 95% (n_days, H, W)
        masks: masques d'invalidité nuages / sol (n_days, H, W)
        dates: liste ou array des dates d'observation en 2018 (strings 'YYYY-MM-DD')
        standing_dieback_map: booléen (H, W), pixels en dépérissement sur pied issus de l'Étape 6
        amplitude_2016: carte d'amplitude annuelle issue de la régression 2016 (H, W)
        autumn_start_date: date à partir de laquelle on teste la réversibilité automnale (défaut: '2018-09-15')
        amplitude_conifer_threshold: seuil d'amplitude séparant résineux (< seuil) et feuillus (> seuil).
                                     Si None, calculé automatiquement via le 60ème percentile.

    Returns:
        dictionnaire contenant :
            - 'drought_map': booléen (H, W), True pour stress sécheresse réversible
            - 'bark_beetle_map': booléen (H, W), True pour attaque de scolytes sur conifères
            - 'deciduous_dieback_map': booléen (H, W), True pour dépérissement irréversible sur feuillus
            - 'classification_map': int (H, W), carte synthétique multi-classes :
                0 = Sain / Non perturbé
                1 = Coupe rase (Clearcut) [réservé pour intégration globale]
                2 = Stress sécheresse réversible (Drought)
                3 = Scolytes sur résineux (Bark Beetles)
                4 = Dépérissement feuillus / autre
            - 'is_conifer': booléen (H, W), masque des peuplements de conifères
            - 'is_reversible': booléen (H, W), masque des pixels ayant récupéré en automne
            - 'amplitude_threshold': float, valeur du seuil d'amplitude retenue
    """
    valid = ~masks & np.isfinite(obs_crswir) & np.isfinite(pred_crswir) & np.isfinite(pred_intervals)
    upper_threshold = pred_crswir + pred_intervals
    anomalies = valid & (obs_crswir > upper_threshold)

    # 1. Analyse de réversibilité automnale
    dates_str = np.array([str(d)[:10] for d in dates])
    autumn_idx = dates_str >= autumn_start_date

    if np.any(autumn_idx):
        autumn_valid = valid[autumn_idx]          # (n_autumn, H, W)
        autumn_anom = anomalies[autumn_idx]       # (n_autumn, H, W)

        n_autumn_valid = np.sum(autumn_valid, axis=0)
        n_autumn_anom = np.sum(autumn_anom, axis=0)

        # Réversible si au moins une date automnale valide et aucune (ou < 25%) anomalie en automne
        with np.errstate(all='ignore'):
            autumn_anom_ratio = np.where(n_autumn_valid > 0, n_autumn_anom / n_autumn_valid, 0.0)

        # Le pixel est réversible s'il n'est plus en anomalie en automne
        is_reversible = standing_dieback_map & (n_autumn_valid > 0) & (autumn_anom_ratio <= 0.25)
    else:
        # En l'absence de dates automnales, on se base sur les anomalies isolées
        total_anom = np.sum(anomalies, axis=0)
        is_reversible = standing_dieback_map & (total_anom <= 1)

    is_irreversible = standing_dieback_map & (~is_reversible)

    # 2. Filtre d'essence via l'amplitude saisonnière 2016
    valid_amp = amplitude_2016[np.isfinite(amplitude_2016)]
    if amplitude_conifer_threshold is None:
        if len(valid_amp) > 0:
            # Séparation basée sur le 60ème percentile de l'amplitude
            amplitude_conifer_threshold = float(np.nanpercentile(valid_amp, 60))
        else:
            amplitude_conifer_threshold = 0.08

    is_conifer = np.isfinite(amplitude_2016) & (amplitude_2016 <= amplitude_conifer_threshold)
    is_deciduous = np.isfinite(amplitude_2016) & (amplitude_2016 > amplitude_conifer_threshold)

    # 3. Cartographie finale
    # - Sécheresse : réhydratation automnale confirmée
    drought_map = standing_dieback_map & is_reversible

    # - Scolytes : mortalité irréversible sur peuplements de conifères
    bark_beetle_map = standing_dieback_map & is_irreversible & is_conifer

    # - Feuillus en dépérissement irréversible (stress hydrique sévère ou autre pathogène)
    deciduous_dieback_map = standing_dieback_map & is_irreversible & is_deciduous

    # Carte catégorielle
    # 0 = Sain, 2 = Sécheresse, 3 = Scolytes, 4 = Dépérissement feuillus
    classification_map = np.zeros(standing_dieback_map.shape, dtype=int)
    classification_map[drought_map] = 2
    classification_map[bark_beetle_map] = 3
    classification_map[deciduous_dieback_map] = 4

    return {
        'drought_map': drought_map,
        'bark_beetle_map': bark_beetle_map,
        'deciduous_dieback_map': deciduous_dieback_map,
        'classification_map': classification_map,
        'is_conifer': is_conifer,
        'is_reversible': is_reversible,
        'amplitude_threshold': amplitude_conifer_threshold,
    }


def classify_forest_disturbances(
    obs_crswir: np.ndarray,
    obs_ndvi: np.ndarray,
    pred_crswir: np.ndarray,
    pred_intervals: np.ndarray,
    masks: np.ndarray,
    dates: np.ndarray,
    amplitude_2016: np.ndarray,
    ndvi_clearcut_threshold: float = 0.30,
    autumn_start_date: str = '2018-09-15',
    amplitude_conifer_threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Chaîne complète de détection et discrimination (Étapes 5, 6 et 7) :
    1. Détection brute du dépérissement (Étape 5)
    2. Discrimination des coupes rases vs dépérissement sur pied (Étape 6)
    3. Discrimination de la sécheresse vs attaques de scolytes sur conifères (Étape 7)

    Codes des classes dans 'final_classification' :
        0: Forêt saine / Non perturbée
        1: Coupe rase (Clearcut)
        2: Stress sécheresse réversible (Drought)
        3: Attaque de scolytes sur conifères (Bark Beetles)
        4: Dépérissement feuillus / autre
    """
    # Étape 5 : Détection du dépérissement
    dieback_res = detect_dieback(obs_crswir, pred_crswir, pred_intervals, masks)

    # Étape 6 : Discrimination coupe rase vs sur pied
    step6_res = distinguish_clearcut_vs_dieback(
        obs_ndvi,
        dieback_res,
        ndvi_clearcut_threshold=ndvi_clearcut_threshold,
    )

    # Étape 7 : Discrimination sécheresse vs scolytes
    step7_res = distinguish_drought_vs_bark_beetle(
        obs_crswir,
        pred_crswir,
        pred_intervals,
        masks,
        dates,
        standing_dieback_map=step6_res['standing_dieback_map'],
        amplitude_2016=amplitude_2016,
        autumn_start_date=autumn_start_date,
        amplitude_conifer_threshold=amplitude_conifer_threshold,
    )

    # Carte finale combinée
    final_class = np.zeros(dieback_res['dieback_map'].shape, dtype=int)
    final_class[step6_res['clearcut_map']] = 1
    final_class[step7_res['drought_map']] = 2
    final_class[step7_res['bark_beetle_map']] = 3
    final_class[step7_res['deciduous_dieback_map']] = 4

    return {
        'dieback_results': dieback_res,
        'clearcut_results': step6_res,
        'drought_beetle_results': step7_res,
        'final_classification': final_class,
        'clearcut_map': step6_res['clearcut_map'],
        'drought_map': step7_res['drought_map'],
        'bark_beetle_map': step7_res['bark_beetle_map'],
        'deciduous_dieback_map': step7_res['deciduous_dieback_map'],
    }
