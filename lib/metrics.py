import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import sklearn.metrics as sklm

from lib.dataset import LightCurveDataset

def report(model, predicted, actual, labels, 
           all_loss, all_error, num_epochs,
           average='macro', zero_division=np.nan, tag=""):
    print("all_loss:", all_loss)
    print("all_error:", all_error)
    output_folder = os.path.join("reports", model)
    if tag:
        output_folder = os.path.join(output_folder, tag)
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
    report = sklm.classification_report(
        actual, predicted, zero_division=zero_division,
        target_names=[str(label) for label in labels]
    )

    print("\n### Comprehensive Classification Report ###")
    print(report)

    # Confusion Matrix
    cm = sklm.confusion_matrix(actual, predicted)
    disp = sklm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues).figure_.savefig(os.path.join(output_folder, "confusion_matrix"))

    # Loss and error curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tick_positions = np.arange(1, num_epochs + 1)
    # Subplot 1: Loss Curve
    train_loss, val_loss = zip(*all_loss)
    axes[0].plot(tick_positions, train_loss, label='Training Loss', color='blue')
    axes[0].plot(tick_positions, val_loss, label='Validation Loss', color='orange')
    axes[0].set_title('Loss Curve Over Epochs', fontsize=16)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Subplot 2: Accuracy Curve
    train_acc, val_acc = zip(*all_error)
    axes[1].plot(tick_positions, train_acc, label='Training Accuracy', color='blue')
    axes[1].plot(tick_positions, val_acc, label='Validation Accuracy', color='orange')
    axes[1].set_title('Accuracy Curve Over Epochs', fontsize=16)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Accuracy', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "training_curves"))


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