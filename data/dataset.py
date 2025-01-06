import pandas as pd
import yaml
import numpy as np
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from scipy.stats import special_ortho_group
import clip

class IMUDataset(Dataset):
    DEFAULT_LOC = ["upperarm", "wrist", "waist", "thigh"]
    def __init__(self, config, augment_round=1, is_train=True):
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

        self.sensor_dimen = 3
        if is_train and augment_round > 0:
            _data_list = []
            # rotation_matrix = special_ortho_group.rvs(self.sensor_dimen, augment_round) # all instance shared the same random rotation matrix
            # if augment_round == 1:
            #     rotation_matrix = np.expand_dims(rotation_matrix, axis=0)
            for data in data_list:
                _data = data.copy()
                instance = np.array(data['imu_input'], dtype=np.float32)
                rotation_matrix = special_ortho_group.rvs(self.sensor_dimen, augment_round) # for each instance, generate random rotation matrix
                if augment_round == 1:
                    rotation_matrix = np.expand_dims(rotation_matrix, axis=0)
                for i in range(augment_round):
                    instance_new = instance.copy().reshape(instance.shape[0], instance.shape[1] // self.sensor_dimen, self.sensor_dimen)
                    for j in range(instance_new.shape[1]):
                        instance_new[:, j, :] = np.dot(instance_new[:, j, :], rotation_matrix[i])
                    instance_new = instance_new.reshape(instance.shape[0], instance.shape[1])
                    _data['imu_input'] = instance_new
                    _data_list.append(_data)
            print(f"before data_list: {len(data_list)}")
            data_list += _data_list

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

        # Load CLIP model
        clip_model, _ = clip.load("ViT-B/32", device="cpu")

        # Precompute location embeddings
        self.location_embeddings = {}
        with torch.no_grad():
            for loc in self.DEFAULT_LOC:
                text = clip.tokenize([loc])
                self.location_embeddings[loc] = clip_model.encode_text(text)
        
        del clip_model

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        imu_data, caption, data_id = sample['imu_input'], sample['output'], sample['data_id']
        location = sample['location']

        imu_input = torch.tensor(imu_data, dtype=torch.float32)
        assert imu_input.shape[1] == 6, f"imu_input shape: {imu_input.shape}"
        label = torch.tensor([self.mapping[caption]], dtype=torch.int8)
        
        location_embedding = self.location_embeddings[location]
        
        return label, imu_input, location_embedding, data_id


class IMUSyncDataset(Dataset):
    DEFAULT_LOC = ["upperarm", "wrist", "waist", "thigh"]
    def __init__(self, config, augment_round=1, is_train=True):
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

        # It can't gurantee the sync data.
        # self.sensor_dimen = 3
        # if is_train and augment_round > 0:
        #     _data_list = []
        #     # rotation_matrix = special_ortho_group.rvs(self.sensor_dimen, augment_round) # all instance shared the same random rotation matrix
        #     # if augment_round == 1:
        #     #     rotation_matrix = np.expand_dims(rotation_matrix, axis=0)
        #     for data in data_list:
        #         _data = data.copy()
        #         instance = np.array(data['imu_input'], dtype=np.float32)
        #         rotation_matrix = special_ortho_group.rvs(self.sensor_dimen, augment_round) # for each instance, generate random rotation matrix
        #         if augment_round == 1:
        #             rotation_matrix = np.expand_dims(rotation_matrix, axis=0)
        #         for i in range(augment_round):
        #             instance_new = instance.copy().reshape(instance.shape[0], instance.shape[1] // self.sensor_dimen, self.sensor_dimen)
        #             for j in range(instance_new.shape[1]):
        #                 instance_new[:, j, :] = np.dot(instance_new[:, j, :], rotation_matrix[i])
        #             instance_new = instance_new.reshape(instance.shape[0], instance.shape[1])
        #             _data['imu_input'] = instance_new
        #             _data_list.append(_data)
        #     print(f"before data_list: {len(data_list)}")
        #     data_list += _data_list

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

        # Load CLIP model
        clip_model, _ = clip.load("ViT-B/32", device="cpu")

        # Precompute location embeddings
        self.location_embeddings = {}
        with torch.no_grad():
            for loc in self.DEFAULT_LOC:
                text = clip.tokenize([loc])
                self.location_embeddings[loc] = clip_model.encode_text(text)
        
        del clip_model

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list.iloc[index]
        imu_data, caption, data_id = sample['imu_input'], sample['output'], sample['data_id']
        location = sample['location']

        imu_input = torch.tensor(imu_data, dtype=torch.float32)
        assert imu_input.shape[1] == 6, f"imu_input shape: {imu_input.shape}"
        
        label = torch.tensor(self.mapping[caption], dtype=torch.int8)
        location_embedding = self.location_embeddings[location]
        
        offset = sample['offset']
        s_id = sample['subject_id']
        
        # Find potential sync data
        matches = self.data_list[
            (self.data_list['offset'] == offset) & 
            (self.data_list['output'] == caption) &
            (self.data_list['subject_id'] == s_id)
            # & (self.data_list['data_id'] != data_id) # exclude current data
        ]
        
        if len(matches) > 0:
            sync_data = matches.sample(1).iloc[0] # it maybe as same as original data
            sync_input = torch.tensor(sync_data['imu_input'], dtype=torch.float32)
            sync_location_embedding = self.location_embeddings[sync_data['location']]
        else:
            sync_input = imu_input.clone()
            sync_location_embedding = location_embedding.clone()
            
        return label, imu_input, location_embedding, data_id, sync_input, sync_location_embedding
