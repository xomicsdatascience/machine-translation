import torch
from torch import nn
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from machine_translation.data import LengthBatchSampler, LineIndexDataset
from transformers import AutoTokenizer

data_directory = 'data'

class MachineTranslationDataModule(pl.LightningDataModule):
    def __init__(self,
                 en_filepath_suffix: str,
                 de_filepath_suffix: str,
                 maximum_length,
                 batch_size,
                 num_training_samples,
                 ):

        super().__init__()
        self.en_filepath_suffix = en_filepath_suffix
        self.de_filepath_suffix = de_filepath_suffix
        self.maximum_length = maximum_length
        self.batch_size = batch_size
        self.num_training_samples = num_training_samples
        self.de_pad_token, self.en_pad_token, self.de_vocab_size, self.en_vocab_size = self.get_tokenizer_values()

    def setup(self, stage=None):
        self.train_dataset = LineIndexDataset(f'{data_directory}/train{self.de_filepath_suffix}', f'{data_directory}/train{self.en_filepath_suffix}', self.num_training_samples)
        self.val_dataset = LineIndexDataset(f'{data_directory}/val{self.de_filepath_suffix}', f'{data_directory}/val{self.en_filepath_suffix}', self.num_training_samples)
        self.test_dataset = LineIndexDataset(f'{data_directory}/test{self.de_filepath_suffix}', f'{data_directory}/test{self.en_filepath_suffix}', self.num_training_samples)

    def train_dataloader(self):
        sampler = torch.utils.data.RandomSampler(self.train_dataset)
        batch_sampler = LengthBatchSampler(sampler, batch_size=self.batch_size, drop_last=False)
        return DataLoader(self.train_dataset, batch_sampler=batch_sampler, collate_fn=self._collate_function)

    def val_dataloader(self):
        sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        batch_sampler = LengthBatchSampler(sampler, batch_size=self.batch_size, drop_last=False)
        return DataLoader(self.val_dataset, batch_sampler=batch_sampler, collate_fn=self._collate_function)

    def test_dataloader(self):
        sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        batch_sampler = LengthBatchSampler(sampler, batch_size=self.batch_size, drop_last=False)
        return DataLoader(self.test_dataset, batch_sampler=batch_sampler, collate_fn=self._collate_function)

    def _collate_function(self, batch):
        input_tensors, expected_output_tensors = zip(*batch)
        src_input_tensor = nn.utils.rnn.pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.de_pad_token,
        )
        output_tensor = nn.utils.rnn.pad_sequence(
            expected_output_tensors,
            batch_first=True,
            padding_value=self.en_pad_token,
        )
        src_input_tensor = src_input_tensor[:, :self.maximum_length]
        output_tensor = output_tensor[:, :self.maximum_length]
        tgt_input_tensor = output_tensor[:, :-1]
        expected_output_tensor = output_tensor[:, 1:]

        src_padding_mask = (src_input_tensor != self.de_pad_token)
        tgt_padding_mask = (tgt_input_tensor != self.en_pad_token)

        return src_input_tensor, tgt_input_tensor, expected_output_tensor, src_padding_mask, tgt_padding_mask

    def get_tokenizer_values(self):
        de_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        return de_tokenizer.convert_tokens_to_ids(de_tokenizer.pad_token), \
            en_tokenizer.convert_tokens_to_ids(en_tokenizer.pad_token), \
            de_tokenizer.vocab_size, en_tokenizer.vocab_size

