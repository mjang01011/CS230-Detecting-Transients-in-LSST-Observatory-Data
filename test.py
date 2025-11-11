import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
from models.rnn import LightCurveRNN
from models.lstm import LightCurveLSTM

def test(args):
    INPUT_SIZE = 1
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = args.num_layers
    BATCH_SIZE = args.batch_size
    SEQUENCE_LENGTH = 100

    test_data = torch.randn(200, SEQUENCE_LENGTH, INPUT_SIZE)
    test_labels = torch.randint(0, 2, (200, 1)).float()

    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == 'rnn':
        model = LightCurveRNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    elif args.model == 'lstm':
        model = LightCurveLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    criterion = nn.BCELoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            test_loss += loss.item()
            predicted = (outputs > 0.5).float()
            test_correct += (predicted == batch_labels).sum().item()
            test_total += batch_labels.size(0)

    test_loss /= len(test_loader)
    test_acc = 100 * test_correct / test_total

    print(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test RNN or LSTM for light curve classification')
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm'], help='Model to use')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()
    test(args)
