import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def report(predicted, actual, average='macro', zero_division=np.nan):
    print(f"Generating report, average={average}, zero_division={zero_division}")
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, average=average, zero_division=zero_division)
    recall = recall_score(actual, predicted, average=average)
    f1 = f1_score(actual, predicted, average=average, zero_division=zero_division)
    
    print("\n### Individual Metric Scores ###")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"Precision Score: {precision:.4f}")
    print(f"Recall Score: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    report = classification_report(actual, predicted, zero_division=zero_division)

    print("\n### Comprehensive Classification Report ###")
    print(report)
