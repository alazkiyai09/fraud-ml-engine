"""Base utilities and helper functions for transformers."""

from typing import Union
import pandas as pd
import numpy as np


def safe_datetime_convert(
    df: pd.DataFrame, datetime_col: str, format: str = None
) -> pd.DataFrame:
    """
    Safely convert a column to datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    datetime_col : str
        Name of datetime column
    format : str, optional
        Datetime format string

    Returns
    -------
    pd.DataFrame
        DataFrame with converted datetime column
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col], format=format, errors="coerce")
    return df


def compute_time_diff_seconds(
    df: pd.DataFrame, datetime_col: str, sort_by: list = None
) -> pd.Series:
    """
    Compute time difference in seconds between consecutive rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    datetime_col : str
        Name of datetime column
    sort_by : list, optional
        Columns to sort by before computing diff

    Returns
    -------
    pd.Series
        Time differences in seconds
    """
    if sort_by:
        df = df.sort_values(sort_by)
    return df[datetime_col].diff().dt.total_seconds()


def handle_unseen_categories(
    series: pd.Series, default: Union[int, float] = 0
) -> pd.Series:
    """
    Handle unseen categorical values by filling with default.

    Parameters
    ----------
    series : pd.Series
        Input series (may contain NaN from unseen categories)
    default : int or float
        Default value to fill

    Returns
    -------
    pd.Series
        Series with NaN values filled
    """
    return series.fillna(default)


def rolling_window_stats(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    datetime_col: str,
    windows: list[tuple],
    agg_funcs: list[str] = ["count", "sum", "mean"],
) -> pd.DataFrame:
    """
    Compute rolling window statistics grouped by a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    group_col : str
        Column to group by (e.g., user_id)
    value_col : str
        Column to aggregate (e.g., amount)
    datetime_col : str
        Datetime column for windowing
    windows : list of tuple
        List of (window_size, unit) tuples, e.g., [(1, 'h'), (24, 'h')]
    agg_funcs : list of str
        Aggregation functions to apply

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling window features
    """
    df = df.sort_values(datetime_col).copy()
    result = pd.DataFrame(index=df.index)

    for window_size, unit in windows:
        window_str = f"{window_size}{unit}"

        for agg_func in agg_funcs:
            if agg_func == "count":
                rolling = df.groupby(group_col)[value_col].rolling(
                    window=window_str, min_periods=1
                ).count()
            elif agg_func == "sum":
                rolling = df.groupby(group_col)[value_col].rolling(
                    window=window_str, min_periods=1
                ).sum()
            elif agg_func == "mean":
                rolling = df.groupby(group_col)[value_col].rolling(
                    window=window_str, min_periods=1
                ).mean()
            elif agg_func == "std":
                rolling = df.groupby(group_col)[value_col].rolling(
                    window=window_str, min_periods=1
                ).std()
            elif agg_func == "max":
                rolling = df.groupby(group_col)[value_col].rolling(
                    window=window_str, min_periods=1
                ).max()
            elif agg_func == "min":
                rolling = df.groupby(group_col)[value_col].rolling(
                    window=window_str, min_periods=1
                ).min()

            feature_name = f"{group_col}_{value_col}_{agg_func}_{window_str}"
            result[feature_name] = rolling.reset_index(level=0, drop=True)

    return result
