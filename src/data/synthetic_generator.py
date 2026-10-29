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

def inject_anomalies(df: pd.DataFrame, spike_ratio: float = 0.05, block_size: int = 50) -> pd.DataFrame:
    """
    Injects synthetic electrical faults as continuous time blocks (bursts)
    into Phase A telemetry to match real-world physical short-circuits.
    
    :param df: Clean telemetry DataFrame
    :param spike_ratio: Target percentage of anomalous rows (~5%)
    :param block_size: Duration of each fault burst in time samples (default: 50)
    :return: DataFrame with block-injected anomalies and updated labels
    """
    df = df.copy()
    num_samples = len(df)
    total_anomalies = int(num_samples * spike_ratio)
    num_blocks = total_anomalies // block_size
    
    # Randomly select starting indices for each continuous fault block
    # Ensure blocks don't overlap or exceed dataset bounds
    valid_starts = np.arange(0, num_samples - block_size, block_size)
    chosen_starts = np.random.choice(valid_starts, size=num_blocks, replace=False)
    
    for start_idx in chosen_starts:
        end_idx = start_idx + block_size
        
        # Draw a uniform spike intensity factor for the duration of the short-circuit
        spike_factor = np.random.uniform(3.5, 5.0)
        
        # Multiply Phase A current over the continuous block duration
        df.loc[start_idx:end_idx - 1, 'current_phase_a'] *= spike_factor
        
        # Mark all timestamps within the fault burst as anomalous
        df.loc[start_idx:end_idx - 1, 'label'] = 1
        
    return df