#!/usr/bin/env python3
"""
Event Detection Algorithms for Financial Time Series

This module provides implementations of various event detection algorithms
used in quantitative finance, particularly for identifying significant 
movements in financial time series data.
"""

from typing import Union
import numpy as np
import pandas as pd
from tqdm import tqdm


def cusum_filter(data: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """
    Extract events using CUSUM (Cumulative Sum) filter.
    
    The CUSUM filter detects events when cumulative positive or negative 
    movements exceed a specified threshold. This is useful for identifying
    significant directional movements in financial time series.
    
    Args:
        data: Time series data (typically log returns or price changes)
        threshold: Threshold for CUSUM filter (positive value)
    
    Returns:
        DatetimeIndex of event timestamps where events were detected
        
    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.008, -0.02], 
        ...                    index=pd.date_range('2024-01-01', periods=4))
        >>> events = cusum_filter(returns, 0.015)
        >>> print(f"Found {len(events)} events")
    """
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    
    if len(data) == 0:
        return pd.DatetimeIndex([])
    
    values = data.values
    timestamps = data.index
    
    # Pre-allocate arrays for performance
    event_mask = np.zeros(len(values), dtype=bool)
    
    # Initialize cumulative sums
    cum_pos, cum_neg = 0.0, 0.0
    
    # Process each value
    for i in tqdm(range(len(values)), desc="Processing CUSUM events", disable=len(values) < 1000):
        # Update cumulative sums
        cum_pos = max(0.0, cum_pos + values[i])
        cum_neg = min(0.0, cum_neg + values[i])
        
        # Check for positive threshold breach
        if cum_pos > threshold:
            event_mask[i] = True
            cum_pos = 0.0
            
        # Check for negative threshold breach  
        if cum_neg < -threshold:
            event_mask[i] = True
            cum_neg = 0.0
    
    return timestamps[event_mask]


class EventProcessor:
    """
    Unified event processing class supporting multiple event detection algorithms.
    
    This class provides a standardized interface for different event detection
    methods, making it easy to switch between algorithms or add new ones.
    """
    
    SUPPORTED_ALGORITHMS = {'cusum'}
    
    def __init__(self, algorithm: str = 'cusum'):
        """
        Initialize the event processor.
        
        Args:
            algorithm: Name of the algorithm to use ('cusum')
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algorithm '{algorithm}' not supported. "
                           f"Supported: {self.SUPPORTED_ALGORITHMS}")
        self.algorithm = algorithm
    
    def extract_events(self, data: pd.Series, threshold: float) -> pd.DatetimeIndex:
        """
        Extract events from time series data using the configured algorithm.
        
        Args:
            data: Time series data
            threshold: Algorithm-specific threshold parameter
            
        Returns:
            DatetimeIndex of detected events
        """
        if self.algorithm == 'cusum':
            return cusum_filter(data, threshold)
        else:
            raise NotImplementedError(f"Algorithm '{self.algorithm}' not implemented")
    
    def get_algorithm_info(self) -> dict:
        """
        Get information about the current algorithm.
        
        Returns:
            Dictionary with algorithm metadata
        """
        if self.algorithm == 'cusum':
            return {
                'name': 'CUSUM Filter',
                'description': 'Detects events when cumulative movements exceed threshold',
                'parameters': ['threshold'],
                'suitable_for': ['log returns', 'price changes', 'directional movements']
            }
        return {}


def validate_event_data(data: Union[pd.Series, pd.DataFrame], 
                       column: str = None) -> pd.Series:
    """
    Validate and extract event detection data from various input formats.
    
    Args:
        data: Input data (Series or DataFrame)
        column: Column name if data is DataFrame
        
    Returns:
        Series suitable for event detection
        
    Raises:
        ValueError: If data is invalid or column not found
    """
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("Column name must be specified for DataFrame input")
        if column not in data.columns:
            available = ', '.join(data.columns)
            raise ValueError(f"Column '{column}' not found. Available: {available}")
        return data[column]
    
    elif isinstance(data, pd.Series):
        if column is not None and column != data.name:
            print(f"Warning: Column '{column}' specified but Series name is '{data.name}'. Using Series as-is.")
        return data
    
    else:
        raise ValueError("Data must be pandas Series or DataFrame")