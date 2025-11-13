import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.dataset import LightCurveDataset
from models.rnn import LightCurveRNN
from models.lstm import LightCurveLSTM

def test(args):
    dataset = LightCurveDataset(
        csv_path=args.data_path,
        max_length=args.max_length,
        use_flux_only=True
    )

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    num_classes = dataset.num_classes
    print(f"Number of classes: {num_classes}")
    print(f"Target mapping: {dataset.target_mapping}")
    print(f"Test objects: {len(dataset.object_ids)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == 'rnn':
        model = LightCurveRNN(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, num_classes=num_classes).to(device)
    elif args.model == 'lstm':
        model = LightCurveLSTM(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    test_correct = 0
    test_total = 0

    test_labels = np.array([])
    pred_labels = np.array([])
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            outputs = model(batch_data)
            predicted = outputs.argmax(dim=1)
            test_correct += (predicted == batch_labels).sum().item()
            test_total += batch_labels.size(0)

            test_labels = np.append(test_labels, batch_labels)
            pred_labels = np.append(pred_labels, predicted)

    print(test_labels)
    print(pred_labels)
    
