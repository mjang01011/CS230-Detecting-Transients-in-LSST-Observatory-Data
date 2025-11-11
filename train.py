import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
from models.rnn import LightCurveRNN
from models.lstm import LightCurveLSTM

def train(args):
    INPUT_SIZE = 1
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = args.num_layers
    LR = args.lr
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    SEQUENCE_LENGTH = 100

    train_data = torch.randn(1000, SEQUENCE_LENGTH, INPUT_SIZE)
    train_labels = torch.randint(0, 2, (1000, 1)).float()
    val_data = torch.randn(200, SEQUENCE_LENGTH, INPUT_SIZE)
    val_labels = torch.randint(0, 2, (200, 1)).float()

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == 'rnn':
        model = LightCurveRNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    elif args.model == 'lstm':
        model = LightCurveLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    if args.save_model:
        torch.save(model.state_dict(), f"{args.model}_model.pth")
        print(f"Model saved to {args.model}_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN or LSTM for light curve classification')
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm'], help='Model to use')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_model', action='store_true', help='Save model after training')

    args = parser.parse_args()
    train(args)
