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