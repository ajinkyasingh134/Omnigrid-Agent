import matplotlib.pyplot as plt
from src.data.synthetic_generator import generate_base_telemetry, inject_anomalies

def plot_telemetry(df, num_points: int = 1000):
    """
    Plots a slice of 3-phase current telemetry and highlights injected fault spikes.
    """
    subset = df.iloc[:num_points]
    
    plt.figure(figsize=(14, 6))
    
    # Plot normal 3-phase current channels
    plt.plot(subset['timestamp'], subset['current_phase_a'], label='Phase A', alpha=0.8, color='#1f77b4')
    plt.plot(subset['timestamp'], subset['current_phase_b'], label='Phase B', alpha=0.8, color='#ff7f0e')
    plt.plot(subset['timestamp'], subset['current_phase_c'], label='Phase C', alpha=0.8, color='#2ca02c')
    
    # Overlay red dots at anomaly locations on Phase A
    anomalies = subset[subset['label'] == 1]
    plt.scatter(anomalies['timestamp'], anomalies['current_phase_a'], 
                color='red', label='Fault Spike (Phase A)', zorder=5, s=30)
    
    plt.title('3-Phase AC Telemetry Stream with Simulated Overcurrent Spikes')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Current (Amperes)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = generate_base_telemetry()
    df_faults = inject_anomalies(df)
    plot_telemetry(df_faults)