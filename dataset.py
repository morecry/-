import os
import multiprocessing as mp
import random

import jieba
import tqdm as tqdm
import torch
from torch.utils.data import Dataset



class MaskDataset(Dataset):
    def __init__(self, file_path, data_source_file, transform, word=7, sample_cnt=None):
        self.transform = transform
        self.word = word

        with open(file_path, encoding='utf-8') as f:
            self.data_source = f.read().strip().split('\n')
        if sample_cnt is not None:
            self.data_source = self.data_source[:sample_cnt]
        # for line in tqdm.tqdm(lines, total=len(lines)):
        #     self.data_source.append(self.transform(line))
        #     if sample_cnt is not None and len(self.data_source) >= sample_cnt:
        #         break
        # torch.save(self.data_source, data_source_file)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        return self.data_source[index]


    def batchify(self, batch):
        inputs_batch, labels_batch = [], []
        for sample in batch:
            sample = self.transform(sample)
            inputs_batch.append(sample[0])
            labels_batch.append(sample[1])
        inputs_batch = torch.tensor(inputs_batch)
        labels_batch = torch.tensor(labels_batch)
        return inputs_batch, labels_batch

class NxtDataset(Dataset):
    def __init__(self, file_path, data_source_file, transform, word=7, sample_cnt=None):
        self.transform = transform
        self.data_source = []
        self.word = word

        if os.path.exists(data_source_file):
            self.data_source = torch.load(data_source_file)
        else:
            pool = []
            with open(file_path, encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
                for line in tqdm.tqdm(lines, total=len(lines)):
                    pool.append(line[0:self.word])
                    pool.append(line[self.word + 1:self.word * 2 + 1])
                for line in tqdm.tqdm(lines, total=len(lines)):
                    pre = line[0:self.word]
                    pst = line[self.word + 1:self.word * 2 + 1]
                    neg = random.choice(pool)
                    tsfm = self.transform(pre, pst, neg)
                    self.data_source.append(tsfm[0])
                    self.data_source.append(tsfm[1])
                    if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                        break
            torch.save(self.data_source, data_source_file)


    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        return self.data_source[index]

    def batchify(self, batch):
        inputs_batch = []
        for sample in batch:
            inputs_batch.append(sample)
        inputs_batch = torch.tensor(inputs_batch)
        return inputs_batch

class MatchDataset(Dataset):
    def __init__(self, file_path, data_source_file, transform, word=7, sample_cnt=None):
        self.transform = transform
        self.data_source = []
        self.word = word

        if os.path.exists(data_source_file):
            self.data_source = torch.load(data_source_file)
        else:
            with open(file_path, encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
                for line in tqdm.tqdm(lines, total=len(lines)):
                    splits = line.strip().split('\t')
                    trs = splits[0]
                    poems = splits[1:]
                    self.data_source.append(self.transform(trs, poems))
                    if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                        break
            torch.save(self.data_source, data_source_file)


    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        return self.data_source[index]


    def batchify(self, batch):
        poem_batch = []
        ch_batch = []
        ch_mask_batch = []
        for sample in batch:
            tag = True
            for poem in sample[2]:
                if len(poem) != 9:
                    tag = False
                    break
            if tag:
                ch_batch.append(sample[0])
                ch_mask_batch.append(sample[1])
                poem_batch.append(sample[2])

        ch_batch = torch.tensor(ch_batch)
        ch_mask_batch = torch.tensor(ch_mask_batch)
        poem_batch = torch.tensor(poem_batch)
        return ch_batch, ch_mask_batch, poem_batch