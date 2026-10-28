import numpy as np
import pandas as pd
def generate_base_telemetry(num_samples: int = 10000, sampling_rate: int = 1000, grid_freq: float = 50.0):
    t = np.linspace(0, num_samples / sampling_rate, num_samples, endpoint=False)
    omega = 2 * np.pi * grid_freq
    noise_amplitude = 0.5
    phase_a = 50.0 * np.sin(omega * t) + np.random.normal(0, noise_amplitude, num_samples)
    phase_b = 50.0 * np.sin(omega * t - (2 * np.pi / 3)) + np.random.normal(0, noise_amplitude, num_samples)
    phase_c = 50.0 * np.sin(omega * t - (4 * np.pi / 3)) + np.random.normal(0, noise_amplitude, num_samples)
    df = pd.DataFrame({
    'timestamp': t,
    'current_phase_a': phase_a,
    'current_phase_b': phase_b,
    'current_phase_c': phase_c,
    'label': 0  # 0 = Normal operating state
    })
    return df

def inject_anomalies(df: pd.DataFrame, spike_ratio: float = 0.05) -> pd.DataFrame:
    """
    Injects synthetic electrical faults (current spikes on Phase A) 
    into the telemetry DataFrame.
    
    :param df: DataFrame containing normal 3-phase telemetry
    :param spike_ratio: Percentage of total rows to corrupt with faults (default: 5%)
    :return: DataFrame with injected anomalies and updated ground-truth labels
    """
    # Create a clean copy to avoid mutating original data in-place
    df = df.copy()
    num_samples = len(df)
    
    # Calculate how many data rows should be marked as anomalous
    num_anomalies = int(num_samples * spike_ratio)
    
    # Randomly pick unique row indices across the dataset to inject faults
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    
    # Generate random spike multiplier factors between 3.5x and 5.0x nominal current
    spike_factors = np.random.uniform(3.5, 5.0, size=num_anomalies)
    
    # Multiply Phase A current at selected indices by the spike factor
    df.loc[anomaly_indices, 'current_phase_a'] *= spike_factors
    
    # Update the ground-truth target label to 1 for these anomalous rows
    df.loc[anomaly_indices, 'label'] = 1
    
    return df