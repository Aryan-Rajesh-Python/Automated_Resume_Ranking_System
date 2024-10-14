import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

class AIOps:
    def __init__(self, data, contamination=0.05):
        self.data = data
        self.contamination = contamination

    def detect_anomalies(self, metrics):
        anomalies = pd.DataFrame()

        # Detect anomalies for each metric
        for metric in metrics:
            model = IsolationForest(contamination=self.contamination)
            self.data[f'{metric}_anomaly'] = model.fit_predict(self.data[[metric]])
            
            # Append detected anomalies
            metric_anomalies = self.data[self.data[f'{metric}_anomaly'] == -1]
            anomalies = pd.concat([anomalies, metric_anomalies])

            # Alert if anomalies are found
            if not metric_anomalies.empty:
                self.send_alert(metric, metric_anomalies)

        return anomalies

    def send_alert(self, metric, anomalies):
        # Simulated alerting function
        print(f"Alert! Anomalies detected in {metric}:")
        print(anomalies)

    def visualize_anomalies(self, metrics):
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            plt.plot(self.data['date'], self.data[metric], label=f'{metric} Metric')
            plt.scatter(self.data[self.data[f'{metric}_anomaly'] == -1]['date'], 
                        self.data[self.data[f'{metric}_anomaly'] == -1][metric], 
                        color='red', label=f'{metric} Anomalies')
            plt.title(f'Anomaly Detection in {metric}')
            plt.xlabel('Date')
            plt.ylabel(f'{metric} Value')
            plt.legend()
            plt.show()

if __name__ == "__main__":
    # Sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Simulate two different metrics with some anomalies
    metric_values_1 = np.random.normal(loc=100, scale=10, size=len(dates))
    metric_values_1[::10] += np.random.normal(loc=50, scale=20, size=len(dates[::10]))  # Introduce anomalies
    
    metric_values_2 = np.random.normal(loc=200, scale=15, size=len(dates))
    metric_values_2[::8] -= np.random.normal(loc=60, scale=30, size=len(dates[::8]))  # Introduce anomalies

    data = pd.DataFrame({'date': dates, 'metric_1': metric_values_1, 'metric_2': metric_values_2})
    
    aiops = AIOps(data, contamination=0.05)

    # Detect anomalies across multiple metrics
    anomalies = aiops.detect_anomalies(['metric_1', 'metric_2'])
    print("Detected Anomalies:")
    print(anomalies)

    # Visualize anomalies for both metrics
    aiops.visualize_anomalies(['metric_1', 'metric_2'])