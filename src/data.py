import os
import json
import csv
import logging
import torch
import src
from src import normalize_text


logger = logging.getLogger(__name__)

def load_passages_csv(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)
        else:
            reader = csv.reader(fin)
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "text": row[1]}
                    passages.append(ex)
    return passages


def load_passages(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)
        else:
            reader = csv.reader(fin, delimiter="\t")
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "text": row[1]}
                    passages.append(ex)
    return passages




class Dataset(torch.utils.data.Dataset):
    def __init__(self, datapaths, normalize=False, global_rank=-1, world_size=-1, maxload=None, training=False):
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

    def __len__(self):
        return len(self.data)

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def __getitem__(self, index):
        example = self.data[index]
        example={"question": self.normalize_fn(example['question']),
                 "gold_answer": self.normalize_fn(example['answer']),
                 "lm_answer": self.normalize_fn(example['llama7b_chat_answer_tempdefault'])}

        return example


class Dataset_squad(torch.utils.data.Dataset):
    def __init__(self, datapaths, normalize=False, global_rank=-1, world_size=-1, maxload=None, training=False):
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

    def __len__(self):
        return len(self.data)

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def __getitem__(self, index):
        example = self.data[index]
        example={"question": self.normalize_fn(example['question']),
                 "gold_answer": self.normalize_fn(example['answer']),
                 "lm_answer": self.normalize_fn(example['t0'])}

        return example
