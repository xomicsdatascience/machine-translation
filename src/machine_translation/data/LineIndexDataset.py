import torch
from torch.utils.data import Dataset
from itertools import islice

class LineIndexDataset(Dataset):
    def __init__(self, input_filepath, expected_output_filepath):
        self.input_filepath = input_filepath
        self.expected_output_filepath = expected_output_filepath
        lengths = []
        with open(input_filepath, 'r') as input_file, open(expected_output_filepath, 'r') as expected_output_file:
            for input_line, expected_output_line in zip(input_file, expected_output_file):

                input_token_count = input_line.count(',')
                output_token_count = expected_output_line.count(',')
                lengths.append(input_token_count + output_token_count)
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        with open(self.input_filepath, 'r') as input_file, \
            open(self.expected_output_filepath, 'r') as expected_output_file:
            input_tensor = [int(token) for token in next(islice(input_file, idx, idx+1)).strip().split(',')]
            line = next(islice(expected_output_file, idx, idx+1)).strip()
            expected_output_tensor = [int(token) for token in line.split(',')]
            return torch.tensor(input_tensor), torch.tensor(expected_output_tensor)

