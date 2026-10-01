import torch

class BatchIndicesIterator:
    def __init__(self, start: int, end: int, batch_size: int, shuffle: bool = True):
        self.start = start
        self.end = end
        if self.start >= self.end:
            raise ValueError(f'The start index {self.start} must be less than the end index {self.end}.')
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = torch.arange(start, end)
        self.num_indices = len(self.indices)
        self.current_index = 0

        if self.shuffle:
            self.indices = self.indices[torch.randperm(self.num_indices)]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.num_indices:
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return batch_indices

    def reset(self):
        """Restart the iterator from the beginning. Will reshuffle if shuffle=True."""
        self.current_index = 0
        if self.shuffle:
            self.indices = self.indices[torch.randperm(self.num_indices)]

    def __len__(self):
        """Returns the number of batches per epoch."""
        return (self.num_indices + self.batch_size - 1) // self.batch_size
