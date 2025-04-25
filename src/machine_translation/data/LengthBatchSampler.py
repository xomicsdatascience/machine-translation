import torch
from torch.utils.data import BatchSampler, Sampler
import torch.distributed as dist
import random


class LengthBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False, dataset=None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Handle both regular and distributed cases
        if hasattr(sampler, 'data_source'):
            self.lengths = getattr(sampler.data_source, 'lengths', None)
        elif dataset is not None:
            self.lengths = dataset.lengths
        else:
            raise ValueError("Either sampler must have data_source with lengths or dataset must be provided")

    def __iter__(self):
        # Get indices from sampler (handles both distributed and non-distributed cases)
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

        # Shuffle batches if using a random sampler
        if not isinstance(self.sampler, torch.utils.data.distributed.DistributedSampler):
            if isinstance(self.sampler, torch.utils.data.RandomSampler):
                random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size