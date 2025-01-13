import torch
from torch.utils.data import Dataset
import mmap

class LineIndexDataset(Dataset):
    def __init__(self, input_filepath, expected_output_filepath, num_training_samples):
        self.input_file = MappedFile(input_filepath)
        self.expected_output_file = MappedFile(expected_output_filepath)
        self.num_training_samples = num_training_samples
        lengths = []
        with open(input_filepath, 'r') as input_file, open(expected_output_filepath, 'r') as expected_output_file:
            for input_line, expected_output_line in zip(input_file, expected_output_file):
                input_token_count = input_line.count(',')
                output_token_count = expected_output_line.count(',')
                lengths.append(input_token_count + output_token_count)
        self.lengths = lengths


    def __len__(self):
        if self.num_training_samples == None:
            return len(self.lengths)
        else:
            return min(len(self.lengths), self.num_training_samples)

    def __getitem__(self, idx):
        input_tensor = [int(token) for token in self.input_file.get_line(idx).strip().split(',')]
        expected_output_tensor = [int(token) for token in self.expected_output_file.get_line(idx).strip().split(',')]
        return torch.tensor(input_tensor), torch.tensor(expected_output_tensor)

class MappedFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.mm = None
        self.line_offsets = []
        self.load()

    def load(self):
        with open(self.file_path, 'r') as file:
            self.mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            offset = 0
            while True:
                newline_pos = self.mm.find(b'\n', offset)
                if newline_pos == -1:
                    break
                self.line_offsets.append(offset)
                offset = newline_pos + 1
            self.line_offsets.append(offset)

    def get_line(self, line_number):
        if line_number < 0 or line_number > len(self.line_offsets):
            raise ValueError("Invalid line number")
        start = self.line_offsets[line_number]
        end = self.line_offsets[line_number + 1]
        return self.mm[start:end].decode('utf-8').strip()
