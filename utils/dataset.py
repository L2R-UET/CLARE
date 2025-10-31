import pandas as pd
from tqdm import tqdm
import pickle as pkl
import json

class Dataset:
    def __init__(self, dataset_root, dataset_name):
        self.trn_csv = pd.read_csv(f"{dataset_root}/{dataset_name}/{dataset_name}_aspect.csv")
        with open(f"{dataset_root}/{dataset_name}/val.pkl", 'rb') as file:
            self.val_csv = pkl.load(file)
        with open(f"{dataset_root}/{dataset_name}/tst.pkl", 'rb') as file:
            self.tst_csv = pkl.load(file)
        self.user_list, self.item_list, self.aspect_list = self.get_data_list()
        self.n_user = len(self.user_list)
        self.n_item = len(self.item_list)
        self.n_aspect = len(self.aspect_list)
        self.user_offset = len(self.aspect_list)
        self.item_offset = len(self.aspect_list) + len(self.user_list)
    
    def get_data_list(self):
        aspect_list = set()
        user_list = set()
        item_list = set()
        
        for index, row in tqdm(self.trn_csv.iterrows()):
            aspects = json.loads(row['aspect'])
            aspects = {key: value for key, value in aspects.items() if type(value) == str}
            for aspect in aspects:
                aspect_list.add(aspect)
            user_list.add(row['uid'])
            item_list.add(row['iid'])
        
        aspect_list = list(aspect_list)
        self.aspect_to_index = {aspect: index for index, aspect in enumerate(aspect_list)}
        self.index_to_aspect = {index: aspect for index, aspect in enumerate(aspect_list)}
        
        user_list = list(user_list)
        self.user_to_index = {user: index for index, user in enumerate(user_list)}
        self.index_to_user = {index: user for index, user in enumerate(user_list)}
        
        item_list = list(item_list)
        self.item_to_index = {item: index for index, item in enumerate(item_list)}
        self.index_to_item = {index: item for index, item in enumerate(item_list)}
        
        aspect_ids = []
        for index, row in tqdm(self.trn_csv.iterrows()):
            aspects = json.loads(row['aspect'])
            aspects = {key: value for key, value in aspects.items() if type(value) == str}
            aspect_ids.append([self.aspect_to_index[aspect] for aspect in aspects])
        
        self.trn_csv['aspect_id'] = aspect_ids
        

        return user_list, item_list, aspect_list