from torch.utils.data import Sampler
import random

class LengthBatchSampler(Sampler):
    def __init__(self, dataset, max_length, shuffle=False, max_tokens=10_000):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.lengths = self.dataset.lengths
        self.shuffle = shuffle
        self.batches = self._setup_batches()

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def _setup_batches(self):
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda i: self.lengths[i])
        batches = []
        current_batch = []
        current_tokens = 0
        for idx in indices:
            sample_length = self.lengths[idx]
            if current_tokens + sample_length > self.max_tokens:
                batches.append(current_batch)
                current_batch = [idx]
                current_tokens = sample_length
            else:
                current_batch.append(idx)
                current_tokens += sample_length
        if current_batch:
            batches.append(current_batch)
        if self.shuffle:
            random.shuffle(batches)
        return batches


