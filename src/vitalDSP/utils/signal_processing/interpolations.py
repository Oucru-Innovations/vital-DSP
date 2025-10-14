import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


def linear_interpolation(intervals):
    """
    Impute missing values using linear interpolation.

    Parameters
    ----------
    intervals : np.array
        The array of RR intervals with NaN values.

    Returns
    -------
    np.array
        The array with missing values imputed using linear interpolation.

    Example
    -------
    >>> linear_interpolation(np.array([0.8, np.nan, 0.82, np.nan, 0.85]))
    """
    rr_series = pd.Series(intervals)
    # Use pandas interpolation with linear method
    return rr_series.interpolate(method="linear", limit_direction="both").to_numpy()


def spline_interpolation(intervals, order=3):
    """
    Impute missing values using spline interpolation.

    Parameters
    ----------
    intervals : np.array
        The array of RR intervals with NaN values.
    order : int, optional
        The order of the spline, defaults to 3 (cubic spline).

    Returns
    -------
    np.array
        The array with missing values imputed using spline interpolation.

    Example
    -------
    >>> spline_interpolation(np.array([0.8, np.nan, 0.82, np.nan, 0.85]), order=3)
    """
    # rr_series = pd.Series(intervals)
    index = np.arange(len(intervals))
    valid = ~np.isnan(intervals)

    # Edge case: if not enough valid points for spline of given order, fall back to linear interpolation
    if valid.sum() < order + 1:
        return linear_interpolation(intervals)

    # Perform spline interpolation
    spline = UnivariateSpline(index[valid], intervals[valid], k=order, s=0)
    return spline(index)


def mean_imputation(intervals):
    """
    Impute missing values by replacing them with the mean of the valid RR intervals.

    Parameters
    ----------
    intervals : np.array
        The array of RR intervals with NaN values.

    Returns
    -------
    np.array
        The array with missing values imputed using the mean of valid intervals.

    Example
    -------
    >>> mean_imputation(np.array([0.8, np.nan, 0.82, np.nan, 0.85]))
    """
    # Avoid "mean of empty slice" warning
    if len(intervals) == 0 or np.all(np.isnan(intervals)):
        return intervals  # Return as-is if no valid data
    mean_value = np.nanmean(intervals)
    return np.where(np.isnan(intervals), mean_value, intervals)


def median_imputation(intervals):
    """
    Impute missing values by replacing them with the median of the valid RR intervals.

    Parameters
    ----------
    intervals : np.array
        The array of RR intervals with NaN values.

    Returns
    -------
    np.array
        The array with missing values imputed using the median of valid intervals.

    Example
    -------
    >>> median_imputation(np.array([0.8, np.nan, 0.82, np.nan, 0.85]))
    """
    # Avoid "All-NaN slice encountered" warning
    if len(intervals) == 0 or np.all(np.isnan(intervals)):
        return intervals  # Return as-is if no valid data
    median_value = np.nanmedian(intervals)
    return np.where(np.isnan(intervals), median_value, intervals)


def forward_fill(intervals):
    """
    Impute missing values by carrying forward the last valid RR interval.

    Parameters
    ----------
    intervals : np.array
        The array of RR intervals with NaN values.

    Returns
    -------
    np.array
        The array with missing values imputed using forward fill.

    Example
    -------
    >>> forward_fill(np.array([0.8, np.nan, 0.82, np.nan, 0.85]))
    """
    rr_series = pd.Series(intervals)
    return rr_series.ffill().bfill().to_numpy()


def backward_fill(intervals):
    """
    Impute missing values by carrying backward the next valid RR interval.

    Parameters
    ----------
    intervals : np.array
        The array of RR intervals with NaN values.

    Returns
    -------
    np.array
        The array with missing values imputed using backward fill.

    Example
    -------
    >>> backward_fill(np.array([0.8, np.nan, 0.82, np.nan, 0.85]))
    """
    rr_series = pd.Series(intervals)
    return rr_series.bfill().ffill().to_numpy()


def rolling_mean_imputation(intervals, window=5):
    """
    Impute missing values using the rolling mean of valid RR intervals.

    Parameters
    ----------
    intervals : np.array
        The array of RR intervals with NaN values.
    window : int, optional
        The window size for the rolling mean. Defaults to 5.

    Returns
    -------
    np.array
        The array with missing values imputed using the rolling mean.

    Example
    -------
    >>> rolling_mean_imputation(np.array([0.8, np.nan, 0.82, np.nan, 0.85]), window=3)
    """
    rr_series = pd.Series(intervals)
    # Apply rolling mean with window size
    intervals_imputed = rr_series.fillna(
        rr_series.rolling(window, min_periods=1).mean()
    )
    # Forward and backward fill for edge cases (start and end NaNs)
    intervals_imputed = intervals_imputed.ffill().bfill()
    return intervals_imputed.to_numpy()
