import numpy as np

def detect_outliers(data, method='iqr', threshold=1.5):
    """
    Detect outliers and their indices from a list or numpy array. 

    Parameters:
    - data (list or np.ndarray): The input data.
    - method (str): 'zscore' or 'iqr' (default: 'iqr').
    - threshold (float): Threshold value for outlier detection.
                         For 'zscore': typically 3.
                         For 'iqr': multiplier for IQR (typically 1.5).

    Returns:
    - outlier_indices (np.ndarray): Indices of outliers.
    - outlier_values (np.ndarray): Values of outliers.
    """
    data = np.array(data)

    if method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        z_scores = (data - mean) / std
        outlier_indices = np.where(np.abs(z_scores) > threshold)[0]

    elif method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]

    else:
        raise ValueError("Method must be 'zscore' or 'iqr'.")

    outlier_values = data[outlier_indices]
    return outlier_indices, outlier_values
