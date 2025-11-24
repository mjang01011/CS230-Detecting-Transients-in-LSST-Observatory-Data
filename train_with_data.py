import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader, random_split
import argparse
from lib.dataset import LightCurveDataset
import lib.metrics as metrics
from models.rnn import LightCurveRNN
from models.lstm import LightCurveLSTM
from models.gru import LightCurveGRU
from models.tcn import LightCurveTCN

def train(args):
    dataset = LightCurveDataset(
        csv_path=args.data_path,
        max_length=args.max_length,
        use_flux_only=True
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_classes = dataset.num_classes
    print(f"Number of classes: {num_classes}")
    print(f"Target mapping: {dataset.target_mapping}")
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.model == 'rnn':
        model = LightCurveRNN(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, num_classes=num_classes).to(device)
    elif args.model == 'lstm':
        model = LightCurveLSTM(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, num_classes=num_classes).to(device)
    elif args.model == 'gru':
        model = LightCurveGRU(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, num_classes=num_classes).to(device)
    elif args.model == 'tcn':
        model = LightCurveTCN(input_size=1, num_classes=num_classes, max_length=args.max_length).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    summary(model, (args.batch_size, args.max_length, 1))

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            predicted = outputs.argmax(dim=1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        epoch_preds = []
        epoch_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                predicted = outputs.argmax(dim=1)

                val_loss += loss.item()
                val_correct += (predicted == batch_labels).sum().item()
                val_total += batch_labels.size(0)

                epoch_preds.extend(predicted)
                epoch_labels.extend(batch_labels)

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        if epoch == args.epochs - 1:
            # Compute final validation report.
            metrics.report(args.model, epoch_preds, epoch_labels)

    if args.save_model:
        torch.save(model.state_dict(), f"{args.model}_model_{args.identifier}.pth")
        print(f"Model saved to {args.model}_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN, LSTM, or GRU on light curve data')
    parser.add_argument('--data_path', type=str, default='data/processed_training.csv', help='Path to processed CSV')
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm', 'gru', 'tcn'], help='Model to use')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=200, help='Max sequence length')
    parser.add_argument('--save_model', action='store_true', help='Save model after training')

    args = parser.parse_args()
    train(args)
