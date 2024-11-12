import torch
from torch import nn
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from machine_translation.data import LineIndexDataset
from transformers import AutoTokenizer

class MachineTranslationDataModule(pl.LightningDataModule):
    def __init__(self,
                 en_filepath_suffix: str,
                 de_filepath_suffix: str,
                 maximum_length=512,
                 batch_size=32,
                 ):

        super().__init__()
        self.en_filepath_suffix = en_filepath_suffix
        self.de_filepath_suffix = de_filepath_suffix
        self.maximum_length = maximum_length
        self.batch_size = batch_size
        self.de_pad_token, self.en_pad_token, self.de_vocab_size, self.en_vocab_size = self.get_tokenizer_values()

    def setup(self, stage=None):
        self.train_dataset = LineIndexDataset(f'data/train{self.de_filepath_suffix}', f'data/train{self.en_filepath_suffix}')
        self.val_dataset = LineIndexDataset(f'data/val{self.de_filepath_suffix}', f'data/val{self.en_filepath_suffix}')
        self.test_dataset = LineIndexDataset(f'data/test{self.de_filepath_suffix}', f'data/test{self.en_filepath_suffix}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self._collate_function, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self._collate_function, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self._collate_function, shuffle=False)

    def _collate_function(self, batch):
        input_tensors, expected_output_tensors = zip(*batch)
        input_tensor = nn.utils.rnn.pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.de_pad_token,
        )
        expected_output_tensor = nn.utils.rnn.pad_sequence(
            expected_output_tensors,
            batch_first=True,
            padding_value=self.en_pad_token,
        )
        input_tensor = input_tensor[:, :self.maximum_length]
        expected_output_tensor = expected_output_tensor[:, :self.maximum_length]

        input_padding_mask = (input_tensor != self.de_pad_token)
        expected_output_padding_mask = (expected_output_tensor != self.en_pad_token)

        return input_tensor, expected_output_tensor, input_padding_mask, expected_output_padding_mask

    def get_tokenizer_values(self):
        de_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        return de_tokenizer.convert_tokens_to_ids(de_tokenizer.pad_token), \
            en_tokenizer.convert_tokens_to_ids(en_tokenizer.pad_token), \
            de_tokenizer.vocab_size, en_tokenizer.vocab_size

