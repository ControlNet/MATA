from __future__ import annotations

import json
import os

import pandas as pd
from rich.progress import track
from torch.utils.data import Dataset


class Refcoco(Dataset):

    def __init__(self, data_root: str, split: str = "testA", max_num: int = 10000000) -> None:
        super().__init__()
        self.data_root = data_root
        # open test dataset
        with open(os.path.join(self.data_root, f"{split}.json")) as refcocotrain:
            ref_coco_train_data = json.load(refcocotrain)

        ref_df = pd.DataFrame(
            columns=['img_name', 'sent_id', 'sub_query', 'ground_true'])

        for img_set in track(ref_coco_train_data):

            # load each data
            img_name = img_set['img_name']

            ground_true = img_set['bbox']

            for sub in img_set['sentences']:

                # sub_query (content/question)
                sub_query = sub['sent']
                # query_id
                sent_id = sub['sent_id']

                new_row_ = pd.DataFrame.from_records(
                    [{'img_name': 'train2014/'+img_name, 'sent_id': sent_id, 'sub_query': sub_query, 'ground_true': ground_true}])
                ref_df = pd.concat([ref_df, new_row_], ignore_index=True)

                if ref_df.shape[0] >= max_num:
                    break
            if ref_df.shape[0] >= max_num:
                break
        self.metadata = ref_df
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        return os.path.join(self.data_root, row.img_name), row.sent_id, row.sub_query, row.ground_true
    
    def __len__(self):
        return len(self.metadata)


class GQA(Dataset):

    def __init__(self, data_root: str, split: str = "testdev", max_num: int = None) -> None:
        super().__init__()
        self.data_root = data_root
        # open test dataset
        assert split in ["train", "val", "test", "testdev"]
        json_path = os.path.join(data_root, f'{split}_balanced_questions.json')
        with open(json_path) as gqa_raw_file:
            gqa_data = json.load(gqa_raw_file)

        qa_df = pd.DataFrame(
            columns=['img_name', 'sent_id', 'sub_query', 'ground_true'])

        sent_id_list = [key for key in gqa_data]

        for sent_id in track(sent_id_list[:max_num], description='Loading GQA dataset'):
            one_set = gqa_data[sent_id]

            # load each data
            img_name = 'gqa_images/' + one_set['imageId']+'.jpg'

            # ground_true
            ground_true = one_set['answer']

            sub_query = one_set['question']

            new_row_ = pd.DataFrame.from_records(
                [{'img_name': img_name, 'sent_id': sent_id, 'sub_query': sub_query, 'ground_true': ground_true}])
            qa_df = pd.concat([qa_df, new_row_], ignore_index=True)
        self.metadata = qa_df
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        return os.path.join(self.data_root, row.img_name), row.sent_id, row.sub_query, row.ground_true
    
    def __len__(self):
        return len(self.metadata)
    
