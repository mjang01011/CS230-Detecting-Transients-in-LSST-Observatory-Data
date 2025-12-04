# CS230-Detecting-Transients-in-LSST-Observatory-Data
CS230 Deep Learning Project

## Models
`models/` folder contains:
- **RNN**: RNN for classification of light curve anomalies
- **LSTM**: LSTM for classification of light curve anomalies
- **GRU**: GRU for classification of light curve anomalies
- **TCN**: https://github.com/paul-krug/pytorch-tcn
- **TS2Vec**: Clone from https://github.com/zhihanyue/ts2vec.git
  ```bash
  cd models
  git clone https://github.com/zhihanyue/ts2vec.git
  ```

## Data
Training and test data sourced from kaggle: https://www.kaggle.com/competitions/PLAsTiCC-2018/data.

Create the following directory structure in your repository:
```
root
  data
    input/
    output/
```

`input` stores kaggle inputs. `output` stores preprocessed data csvs.



### Preprocessing
```
# Pre-process training data
python3 main.py preprocess --meta_filename=training_set_metadata --raw_filename=training_set --processed_filename=processed_training

# Pre-process subset training data
python3 main.py preprocess --meta_filename=training_set_metadata --raw_filename=training_set --processed_filename=processed_training_42_65_90 --targets 42 65 90

python3 main.py preprocess --meta_filename=training_set_metadata --raw_filename=training_set --processed_filename=processed_training_16_65 --targets 16 65


# Pre-process test data
python3 main.py preprocess --meta_filename=test_set_metadata --raw_filename=test_set_sample --processed_filename=processed_test
```

## Training
```
python3 train_with_data.py --model=rnn

python3 main.py train --model=rnn --data_path=data/output/processed_training_42_65_90.csv
python3 main.py train --model=rnn --data_path=data/output/processed_training_16_65.csv

```

### Use more parameters
```
python3 main.py train --model=tcn --data_path=data/output/processed_training_16_65.csv --use_flux_only=False
```

## Testing
```
python3 main.py test --model=rnn --data_path=data/output/processed_test.csv
```