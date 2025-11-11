import argparse

from lib.preprocessing import full_preprocess
from train_with_data import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN or LSTM on light curve data')
    parser.add_argument("command", help="The command to run [preprocess | train].")
    parser.add_argument('--data_path', type=str, default='data/processed_training.csv', help='Path to processed CSV')
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm'], help='Model to use')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=200, help='Max sequence length')
    parser.add_argument('--save_model', action='store_true', help='Save model after training')

    args = parser.parse_args()
    if args.command == "preprocess":
        full_preprocess()
    elif args.command == "train":
        train(args)
    else:
        raise NotImplementedError("Command not found.")
    