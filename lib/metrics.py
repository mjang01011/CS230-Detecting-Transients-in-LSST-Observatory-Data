import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sklm
from torch.utils.data import DataLoader, random_split

from lib.dataset import LightCurveDataset

def report(model, predicted, actual, average='macro', zero_division=np.nan):
    output_folder = os.path.join("reports", model)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Generating report, average={average}, zero_division={zero_division}")
    accuracy = sklm.accuracy_score(actual, predicted)
    precision = sklm.precision_score(actual, predicted, average=average, zero_division=zero_division)
    recall = sklm.recall_score(actual, predicted, average=average)
    f1 = sklm.f1_score(actual, predicted, average=average, zero_division=zero_division)
    
    print("\n### Individual Metric Scores ###")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"Precision Score: {precision:.4f}")
    print(f"Recall Score: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    report = sklm.classification_report(actual, predicted, zero_division=zero_division)

    print("\n### Comprehensive Classification Report ###")
    print(report)

    # Confusion Matrix
    cm = sklm.confusion_matrix(actual, predicted)
    disp = sklm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    disp.plot().figure_.savefig(os.path.join(output_folder, "confusion_matrix"))


def check_class_distribution(
    dataset: LightCurveDataset,
    figure_name: str,
):
    # check the distribution of classes in a dataset, output results to a bar chart.
    class_count = {}
    for target_class in range(dataset.num_classes):
        class_count[target_class] = 0
    print(dataset.num_classes)
    for _, obj in dataset.data.items():
        class_count[obj['target']] += 1
    print(class_count)

    plt.bar(class_count.keys(), class_count.values(), color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Quantity')
    plt.title('Class Distribution')
    plt.savefig(figure_name)

def input_analysis(args):
    dataset = LightCurveDataset(
        csv_path=args.data_path,
        max_length=args.max_length,
        use_flux_only=True
    )
    check_class_distribution(dataset, "all_data")