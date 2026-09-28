"""Batch sampler that puts the same number of images of every class in each batch.

Batch size 32 with the 8 classes of src.dataset.CLASSES -> exactly 4 images per class per batch.

Per epoch (random.Random(seed + epoch), so each epoch is deterministic but different):
- each class gets a shuffled copy of its indices; 4 are taken per batch, and when a class runs
  out its indices are reshuffled and reused (sampling with replacement for small classes);
- the number of batches is fixed from outside (num_batches), e.g. ceil(len(train_subset) / 32)
  so that one epoch has as many batches as a standard shuffled DataLoader;
- the 32 indices of a batch are shuffled, so class order inside a batch is not fixed.

Used with DataLoader(dataset, batch_sampler=sampler) - no batch_size / shuffle / sampler arguments.
Call set_epoch(epoch) before each epoch.
"""

import random

from torch.utils.data import Sampler

from src.dataset import CLASSES

BATCH_SIZE = 32


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, num_batches, batch_size=BATCH_SIZE, seed=42):
        """labels: class index (0..7) of every item of the train dataset, in dataset order."""
        if batch_size != BATCH_SIZE:
            raise ValueError(f"batch_size must be {BATCH_SIZE}, got {batch_size}")
        if not isinstance(num_batches, int) or num_batches < 1:
            raise ValueError(f"num_batches must be a positive integer, got {num_batches}")
        num_classes = len(CLASSES)
        bad = sorted({l for l in labels if not (isinstance(l, int) and 0 <= l < num_classes)})
        if bad:
            raise ValueError(f"labels outside the {num_classes} classes: {bad}")

        self.class_indices = [[i for i, l in enumerate(labels) if l == c] for c in range(num_classes)]
        missing = [CLASSES[c] for c, idx in enumerate(self.class_indices) if not idx]
        if missing:
            raise ValueError(f"classes missing from the train subset: {missing}")

        self.per_class = batch_size // num_classes     # 4
        self.num_batches = num_batches
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = [[] for _ in self.class_indices]
        for _ in range(self.num_batches):
            batch = []
            for c, indices in enumerate(self.class_indices):
                while len(pools[c]) < self.per_class:   # a class with < 4 images needs several refills
                    refill = list(indices)
                    rng.shuffle(refill)
                    pools[c] += refill
                batch += pools[c][:self.per_class]
                pools[c] = pools[c][self.per_class:]
            rng.shuffle(batch)
            yield batch
