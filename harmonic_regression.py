import numpy as np
from typing import Dict
from pandas import to_datetime
from scipy.stats import t as t_dist


REF_DAY = to_datetime("2015-01-01").to_datetime64()


def fit_harmonic_model(
    ts_days: np.ndarray, 
    ts_values: np.ndarray,
    ts_masks: np.ndarray):
    """
    Models periodic vegetation index.

    Args:
        ts_days: array of days, encoded in integer value from a reference date, of shape (n_days,)
        ts_values: array of the time series values in floats, of shape (n_days,)
        ts_masks: array of invalid pixels, of shape (n_days,)

    Returns:
        dictionary containing:
            - a key 'coeff' mapping to the coefficients of the linear model
            - a key 'std_hat' mapping to the estimated standard deviation of the residual errors
            - a key 'XTX' mapping to the Gram matrix of the matrix of input variables
            - a key 'df' mapping to the degrees of freedom of the estimated model
    """
    # ts_values peut être 1D (n_days,) pour un seul pixel
    # ou 3D (n_days, height, width) pour tous les pixels.
    # ts_masks : True = pixel invalide à ignorer.
    spatial = ts_values.ndim == 3
    if spatial:
        n_days, height, width = ts_values.shape
        # Aplatir les dimensions spatiales → (n_days, n_pixels)
        Y = ts_values.reshape(n_days, -1)
        M = ts_masks.reshape(n_days, -1)
    else:
        n_days = ts_values.shape[0]
        Y = ts_values[:, np.newaxis]  # (n_days, 1)
        M = ts_masks[:, np.newaxis]

    n_pixels = Y.shape[1]
    p = 5  # nombre de termes harmoniques

    # Construire la matrice de design X : (n_days, 5)
    X = np.array([compute_harmonic_terms(d) for d in ts_days])  # (n_days, 5)

    # Pour chaque pixel, on résout les moindres carrés en ignorant les NaN/masqués.
    coeff = np.full((p, n_pixels), np.nan)
    std_hat = np.full(n_pixels, np.nan)
    XTX = np.full((n_pixels, p, p), np.nan)
    df = np.full(n_pixels, np.nan)

    for j in range(n_pixels):
        valid = ~M[:, j] & np.isfinite(Y[:, j])
        n_valid = valid.sum()
        if n_valid <= p:
            continue

        X_valid = X[valid]          # (n_valid, 5)
        y_valid = Y[valid, j]      # (n_valid,)

        gram = X_valid.T @ X_valid  # (5, 5)
        try:
            gram_inv = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            continue

        beta = gram_inv @ (X_valid.T @ y_valid)   # (5,)
        residuals = y_valid - X_valid @ beta
        dof = n_valid - p
        sigma = np.sqrt(np.sum(residuals ** 2) / dof)

        coeff[:, j] = beta
        std_hat[j] = sigma
        XTX[j] = gram
        df[j] = dof

    if spatial:
        coeff = coeff.reshape(p, height, width)
        std_hat = std_hat.reshape(height, width)
        XTX = XTX.reshape(height, width, p, p)
        df = df.reshape(height, width)

    return {
        'coeff': coeff,
        'std_hat': std_hat,
        'XTX': XTX,
        'df': df,
    }


def pred_harmonic_model(
    ts_days: np.ndarray,
    harmonic_model: Dict,
    confidence_level: float = 0.95,
    reshape_spatial_dims: bool = True
    ):
    """
    Prediction of a vegetation index by a harmonic model.

    Args:
        ts_days: array of days, encoded in integer value from a reference date, of shape (n_days,)
        harmonic_model: estimated model in a dictionary format, as given by fit_harmonic_model
        confidence_level: level of confidence for the confidence and prediction intervals

    Returns:
        dictionary containing:
            - a key 'ts_hat' mapping to the predictions
            - a key 'prediction_interval' mapping to the prediction interval of new observations
    """
    coeff = harmonic_model['coeff']       # (5,) ou (5, H, W)
    std_hat = harmonic_model['std_hat']   # scalaire ou (H, W)
    gram = harmonic_model['XTX']          # (5,5) ou (H, W, 5, 5)
    dof = harmonic_model['df']            # scalaire ou (H, W)

    spatial = coeff.ndim == 3
    p = 5
    n_days = len(ts_days)

    # Matrice de design pour les nouvelles dates
    X_new = np.array([compute_harmonic_terms(d) for d in ts_days])  # (n_days, 5)

    if spatial:
        height, width = coeff.shape[1], coeff.shape[2]
        # Prédictions : X_new @ coeff → (n_days, H, W)
        # coeff est (5, H, W), on le reshape en (5, H*W)
        coeff_flat = coeff.reshape(p, -1)         # (5, n_pixels)
        ts_hat_flat = X_new @ coeff_flat           # (n_days, n_pixels)

        # Intervalle de prédiction par pixel
        std_flat = std_hat.reshape(-1)             # (n_pixels,)
        dof_flat = dof.reshape(-1)                 # (n_pixels,)
        gram_flat = gram.reshape(-1, p, p)         # (n_pixels, 5, 5)

        pred_half = np.full((n_days, std_flat.shape[0]), np.nan)
        for j in range(std_flat.shape[0]):
            if np.isnan(std_flat[j]) or np.isnan(dof_flat[j]) or dof_flat[j] < 1:
                continue
            try:
                gram_inv = np.linalg.inv(gram_flat[j])
            except np.linalg.LinAlgError:
                continue
            t_crit = t_dist.ppf((1 + confidence_level) / 2, dof_flat[j])
            # Variance de prédiction pour chaque date :
            # Var(y_new) = σ² (1 + x'(X'X)⁻¹x)
            for i in range(n_days):
                x = X_new[i]  # (5,)
                var_pred = std_flat[j] ** 2 * (1 + x @ gram_inv @ x)
                pred_half[i, j] = t_crit * np.sqrt(var_pred)

        if reshape_spatial_dims:
            ts_hat = ts_hat_flat.reshape(n_days, height, width)
            prediction_interval = pred_half.reshape(n_days, height, width)
        else:
            ts_hat = ts_hat_flat
            prediction_interval = pred_half
    else:
        # Cas 1D (un seul pixel)
        ts_hat = X_new @ coeff  # (n_days,)

        t_crit = t_dist.ppf((1 + confidence_level) / 2, dof)
        try:
            gram_inv = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            prediction_interval = np.full(n_days, np.nan)
            return {'ts_hat': ts_hat, 'prediction_interval': prediction_interval}

        pred_half = np.zeros(n_days)
        for i in range(n_days):
            x = X_new[i]
            var_pred = std_hat ** 2 * (1 + x @ gram_inv @ x)
            pred_half[i] = t_crit * np.sqrt(var_pred)

        prediction_interval = pred_half

    return {
        'ts_hat': ts_hat,
        'prediction_interval': prediction_interval,
    }


def dates_to_days(dates):
    """
    Converts datetime objects to days since reference date.

    Args:
        dates : array-like of datetime-like
        Input dates to convert.

    Returns:
        Array of integers representing days since REF_DAY (2015-01-01).
    """
    try:
        days = (dates - REF_DAY).astype("timedelta64[D]").astype(int)
    except ValueError as e:
        # .astype("timedelta64[D]") not supported anymore by pandas series
        # https://stackoverflow.com/a/76363956
        # https://github.com/pandas-dev/pandas/blob/8cf4ab491b06274f555d1ea305c172354208747a/doc/source/whatsnew/v2.0.0.rst#disallow-astype-conversion-to-non-supported-datetime64timedelta64-dtypes
        pd_msg = "Supported resolutions are 's', 'ms', 'us', 'ns'"
        if pd_msg in str(e):
            days = (dates - REF_DAY).days
        else:
            raise e

    return days


def compute_harmonic_terms(DateAsNumber: int, T: float = 365.25):
    """
    Computes harmonic terms for a given date.

    Args:
        DateAsNumber : number of days since reference date (2015-01-01).
        T: periodicity of the signal to model.

    Returns:
        Array of shape (5,) containing harmonic terms.
    """
    omega = 2 * np.pi * DateAsNumber / T
    return np.array([
        1.0,              # intercept (a0)
        np.cos(omega),    # cos annuel (a1)
        np.sin(omega),    # sin annuel (b1)
        np.cos(2 * omega),  # cos semi-annuel (a2)
        np.sin(2 * omega),  # sin semi-annuel (b2)
    ])