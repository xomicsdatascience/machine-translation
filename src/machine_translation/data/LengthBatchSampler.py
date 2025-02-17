import torch
from torch.utils.data import BatchSampler, Sampler
import random


class LengthBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.lengths = getattr(sampler.data_source, 'lengths', None)
        if self.lengths is None:
            raise ValueError("Dataset must have a 'lengths' attribute")

    def __iter__(self):
        indices = list(self.sampler)
        # Sort indices by sequence length
        indices.sort(key=lambda i: self.lengths[i])

        batches = []
        batch = []

        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)

        if isinstance(self.sampler, torch.utils.data.RandomSampler):
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size