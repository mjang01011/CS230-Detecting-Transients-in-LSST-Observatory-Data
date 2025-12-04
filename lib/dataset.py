import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class LightCurveDataset(Dataset):
    def __init__(self, csv_path, max_length=200, use_flux_only=True):
        """
        Dataset for light curve data grouped by object_id

        Args:
            csv_path: path to processed CSV file
            max_length: maximum sequence length (pad/truncate)
            use_flux_only: if True, use only flux, else use mjd, flux, flux_err
        """
        df = pd.read_csv(csv_path)
        self.max_length = max_length
        self.use_flux_only = use_flux_only

        unique_targets = sorted(df['target'].unique())
        self.target_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_targets)}
        self.ordered_labels = [item[0] for item in sorted(self.target_mapping.items(), key=lambda item: item[1])]
        self.num_classes = len(unique_targets)
        print(self.target_mapping)
        self.object_ids = df['object_id'].unique()
        self.data = {}

        for obj_id in self.object_ids:
            obj_data = df[df['object_id'] == obj_id].sort_values('t_centered')
            t_centered = obj_data['t_centered'].values
            passband = obj_data['passband'].values
            flux = obj_data['flux'].values
            flux_err = obj_data['flux_err'].values
            detected = obj_data['detected'].values

            target = obj_data['target'].iloc[0]
            target = self.target_mapping[target]

            self.data[obj_id] = {
                't_centered': t_centered,
                'flux': flux,
                'flux_err': flux_err,
                'passband': passband,
                'detected': detected,
                'target': target
            }

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        obj_id = self.object_ids[idx]
        obj_data = self.data[obj_id]

        t_centered = obj_data['t_centered']
        flux = obj_data['flux']
        flux_err = obj_data['flux_err']
        passband = obj_data['passband']
        detected = obj_data['detected']
        target = obj_data['target']

        if self.use_flux_only:
            sequence = flux
        else:
            sequence = np.stack([t_centered, flux, flux_err, passband, detected], axis=1)

        if len(sequence) > self.max_length:
            # sequence = sequence[:self.max_length]
            mid_idx = len(sequence) // 2
            sequence = sequence[mid_idx - self.max_length // 2 : mid_idx + self.max_length // 2]
        elif len(sequence) < self.max_length:
            if self.use_flux_only:
                padding = np.zeros(self.max_length - len(sequence))
                sequence = np.concatenate([sequence, padding])
            else:
                padding = np.zeros((self.max_length - len(sequence), 2))
                sequence = np.concatenate([sequence, padding])

        if self.use_flux_only:
            sequence = sequence.reshape(-1, 1)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.long)

        return sequence, target
