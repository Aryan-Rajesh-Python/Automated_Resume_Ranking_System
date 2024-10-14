import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

class AIOps:
    def __init__(self, data):
        self.data = data

    def detect_anomalies(self):
        # Using Isolation Forest for anomaly detection
        model = IsolationForest(contamination=0.05)
        self.data['anomaly'] = model.fit_predict(self.data[['metric']])
        anomalies = self.data[self.data['anomaly'] == -1]
        
        # Alert if anomalies are found
        if not anomalies.empty:
            self.send_alert(anomalies)

        return anomalies

    def send_alert(self, anomalies):
        # Simulated alerting function
        print("Alert! Anomalies detected:")
        print(anomalies)

    def visualize_anomalies(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['date'], self.data['metric'], label='Metric')
        plt.scatter(self.data[self.data['anomaly'] == -1]['date'], 
                    self.data[self.data['anomaly'] == -1]['metric'], color='red', label='Anomalies')
        plt.title('Anomaly Detection in Metrics')
        plt.xlabel('Date')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    metric_values = np.random.normal(loc=100, scale=10, size=len(dates))
    metric_values[::10] += np.random.normal(loc=50, scale=20, size=len(dates[::10]))  # Introduce anomalies

    data = pd.DataFrame({'date': dates, 'metric': metric_values})
    aiops = AIOps(data)

    # Detect anomalies
    anomalies = aiops.detect_anomalies()
    print("Detected Anomalies:")
    print(anomalies)

    # Visualize anomalies
    aiops.visualize_anomalies()