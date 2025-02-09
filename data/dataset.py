import pandas as pd
import yaml
import numpy as np
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from scipy.stats import special_ortho_group
import pickle

def random_rotation(data: np.ndarray) -> np.ndarray:
    sensor_dim = 3
    rotation_matrix = special_ortho_group.rvs(sensor_dim)
    data_new = data.copy().reshape(data.shape[0], data.shape[1] // sensor_dim, sensor_dim)
    for i in range(data_new.shape[1]):
        data_new[:, i, :] = np.dot(data_new[:, i, :], rotation_matrix)
    data_new = data_new.reshape(data.shape[0], data.shape[1])
    return data_new

class IMUDataset(Dataset):
    DEFAULT_LOC = ["upperarm", "wrist", "waist", "thigh"]
    def __init__(self, config, is_train=True, is_rotated=False):
        self.is_train = is_train
        self.is_rotated = is_rotated

        data_list = []
        if isinstance(config, str):
            config: Dict = yaml.safe_load(open(config))['TRAIN' if is_train else 'TEST']
        else:
            config: Dict = config
        paths = config['META']
        loc: List[str] = config.get('LOC', self.DEFAULT_LOC)
        for meta_path in paths:
            df = pd.read_json(meta_path, orient='records')
            meta_l = df[df['location'].isin(loc)].to_dict('records')
            print(f"{meta_path}: len {len(meta_l)}")
            data_list += meta_l

        self.data_list = data_list
        print(f"total length: {len(self)}")
        ['downstairs', 'jog', 'lie', 'sit', 'stand', 'upstairs', 'walk']
        self.mapping = {
            'downstairs': 0,
            'jog': 1,
            'lie': 2,
            'sit': 3,
            'stand': 4,
            'upstairs': 5,
            'walk': 6
        }

        with open('/home/wjdu/ctx_aware_pamap2/pamap2_embs.pkl', 'rb') as f:
            embs = pickle.load(f)
        self.embs = {}
        for l in loc:
            self.embs[l] = {}
            for u_id, embeddings in embs[l].items():
                self.embs[l][u_id] = embeddings.requires_grad_(False)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        imu_data = np.array(sample['imu_input'])
        caption, data_id = sample['output'], sample['data_id']
        location = sample['location']
        user_id = sample['subject_id']

        if self.is_rotated:
            imu_data = random_rotation(imu_data)
        
        imu_input = torch.tensor(imu_data, dtype=torch.float32)
        label = torch.tensor([self.mapping[caption]], dtype=torch.int8)
        
        ctx_emb = self.embs[location][user_id]
        return label, imu_input, ctx_emb, data_id


class IMUSyncDataset(Dataset):
    DEFAULT_LOC = ["upperarm", "wrist", "waist", "thigh"]
    def __init__(self, config, is_train=True, is_rotated=False):
        self.is_train = is_train
        self.is_rotated = is_rotated

        data_list = []
        if isinstance(config, str):
            config: Dict = yaml.safe_load(open(config))['TRAIN' if is_train else 'TEST']
        else:
            config: Dict = config
        paths = config['META']
        loc: List[str] = config.get('LOC', self.DEFAULT_LOC)
        for meta_path in paths:
            df = pd.read_json(meta_path, orient='records')
            meta_l = df[df['location'].isin(loc)].to_dict('records')
            print(f"{meta_path}: len {len(meta_l)}")
            data_list += meta_l

        self.data_list = pd.DataFrame(data_list)
        print(f"total length: {len(self)}")
        ['downstairs', 'jog', 'lie', 'sit', 'stand', 'upstairs', 'walk']
        self.mapping = {
            'downstairs': 0,
            'jog': 1,
            'lie': 2,
            'sit': 3,
            'stand': 4,
            'upstairs': 5,
            'walk': 6
        }

        self.location_embs = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list.iloc[index]
        origin_data = np.array(sample['imu_input'])
        caption, data_id = sample['output'], sample['data_id']
        location = sample['location']

        location_emb = self.location_embs[location]
        label = torch.tensor(self.mapping[caption], dtype=torch.int8)

        offset = sample['offset']
        s_id = sample['subject_id']
        # Find potential sync data
        matches = self.data_list[
            # (self.data_list['offset'] == offset) & 
            (self.data_list['output'] == caption) &
            (self.data_list['subject_id'] == s_id) &
            (self.data_list['data_id'] != data_id) # exclude current data
        ]
        
        if len(matches) > 0:
            sync_sample = matches.sample(1).iloc[0] # it maybe as same as original data
            sync_data = np.array(sync_sample['imu_input'])
            if self.is_rotated:
                sync_data = random_rotation(sync_data)
            sync_input = torch.tensor(sync_data, dtype=torch.float32)
            sync_location_emb = self.location_embs[sync_sample['location']]
        else:
            sync_input = torch.tensor(origin_data, dtype=torch.float32)
            sync_location_emb = location_emb.clone()
        
        if self.is_rotated:
            imu_data = random_rotation(origin_data)

        imu_input = torch.tensor(imu_data, dtype=torch.float32)            
        return label, imu_input, location_emb, data_id, sync_input, sync_location_emb
