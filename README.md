# CS230-Detecting-Transients-in-LSST-Observatory-Data
CS230 Deep Learning Project

## Models
`models/` folder contains:
- **RNN**: RNN for binary classification of light curve anomalies
- **LSTM**: LSTM for binary classification of light curve anomalies
- **TCN**: https://github.com/paul-krug/pytorch-tcn
- **TS2Vec**: Clone from https://github.com/zhihanyue/ts2vec.git
  ```bash
  cd models
  git clone https://github.com/zhihanyue/ts2vec.git
  ```

## Data
Create the following directory structure in your repository:
```
root
  data
    input/
    output/
```

`input` stores kaggle inputs. `output` stores output csv and models.

### Preprocessing
```
python3 main.py preprocess
```

## Training
```
python3 train_with_data.py --model=rnn
```