import torch
from torch.utils.data import Dataset
from itertools import islice

class LineIndexDataset(Dataset):
    def __init__(self, input_filepath, expected_output_filepath):
        self.input_filepath = input_filepath
        self.expected_output_filepath = expected_output_filepath

    def __getitem__(self, idx):
        with open(self.input_filepath, 'r') as input_file, \
            open(self.expected_output_filepath, 'r') as expected_output_file:
            input_tensor = [int(token) for token in next(islice(input_file, idx, idx+1)).strip().split(',')]
            expected_output_tensor = [int(token) for token in next(islice(expected_output_file, idx, idx+1)).strip().split(',')]
            return torch.tensor(input_tensor), torch.tensor(expected_output_file)

