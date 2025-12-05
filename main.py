import argparse
from datetime import datetime

from lib.preprocessing import full_preprocess
from train_with_data import train
from test import test
import lib.metrics as metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN or LSTM on light curve data')
    parser.add_argument("command", help="The command to run [preprocess | train].")
    parser.add_argument('--data_path', type=str, default='data/output/processed_training.csv', help='Path to processed CSV')
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm', 'gru', 'tcn'], help='Model to use')
    parser.add_argument('--identifier', type=str, default=f'{datetime.now().timestamp()}', help='Model identifier')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=200, help='Max sequence length')
    parser.add_argument('--save_model', type=bool, default=True, help='Save model after training')
    parser.add_argument('--use_flux_only', type=bool, default=True, help='Only use flux as training parameter')
    parser.add_argument('--tag', type=str, default='', help='Label to separate models')
    parser.add_argument('--kernel_size', type=int, default=3, help='Number of epochs')

    parser.add_argument('--model_path', type=str, default='rnn_model_1762911982.909752.pth')

    parser.add_argument('--meta_filename', type=str, default='training_set_metadata')
    parser.add_argument('--raw_filename', type=str, default='training_set')
    parser.add_argument('--processed_filename', type=str, default='processed_training')
    parser.add_argument('--targets', type=int, nargs='*', default=[], help='Classes to keep')

    args = parser.parse_args()
    if args.command == "preprocess":
        full_preprocess(args)
    elif args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)
    elif args.command == "input_analysis":
        metrics.input_analysis(args)
    else:
        raise NotImplementedError("Command not found.")
    