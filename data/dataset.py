import os
import pandas as pd
import yaml
import numpy as np
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from scipy.stats import special_ortho_group
from transformers import AutoTokenizer, AutoModelForMaskedLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
bert_config = "answerdotai/ModernBERT-base"

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

        tokenizer = AutoTokenizer.from_pretrained(bert_config)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_config)
        bert_model.cuda()
        bert_model.eval()
    
        # Precompute embeddings using BERT
        user_info = {
            1: "a male aged 27 years, with a height of 182 cm and a weight of 83 kg, collected on ",
            2: "a female aged 25 years, with a height of 169 cm and a weight of 78 kg, collected on ",
            3: "a male aged 31 years, with a height of 187 cm and a weight of 92 kg, collected on ",
            4: "a male aged 24 years, with a height of 194 cm and a weight of 95 kg, collected on ",
            5: "a male aged 26 years, with a height of 180 cm and a weight of 73 kg, collected on ",
            6: "a male aged 26 years, with a height of 183 cm and a weight of 69 kg, collected on ",
            7: "a male aged 23 years, with a height of 173 cm and a weight of 86 kg, collected on ",
            8: "a male aged 32 years, with a height of 179 cm and a weight of 87 kg, collected on ",
            9: "a male aged 31 years, with a height of 168 cm and a weight of 65 kg, collected on "
        }
        self.embs = {l: {} for l in loc}
        with torch.no_grad():
            for l in loc:
                self.embs[l] = {}
                for u_id, u_info in user_info.items():
                    text = u_info + l
                    inputs = tokenizer(text, return_tensors="pt").to('cuda')
                    outputs = bert_model(**inputs, output_hidden_states=True)
                    # self.user_embs[u_id] = outputs.hidden_states[-1][:, 0].to('cpu') # CLS token
                    self.embs[l][u_id] = outputs.hidden_states[-1].to('cpu') # sentence embedding
        
        del bert_model
        torch.cuda.empty_cache()

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

        tokenizer = AutoTokenizer.from_pretrained(bert_config)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_config)
        bert_model.cuda()
        bert_model.eval()
        
        # Precompute location embeddings using BERT
        self.location_embs = {}
        with torch.no_grad():
            for l in loc:
                inputs = tokenizer(l, return_tensors="pt")
                outputs = bert_model(**inputs, output_hidden_states=True)
                # self.location_embs[l] = outputs.hidden_states[-1][:, 0].to('cpu')
                self.location_embs[l] = outputs.hidden_states[-1].to('cpu')
        
        del bert_model

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
