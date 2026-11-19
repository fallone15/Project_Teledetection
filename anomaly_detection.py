import numpy as np
from typing import Dict


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
    dieback_map = anomaly_count > 0

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
