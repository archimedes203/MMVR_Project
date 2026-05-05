"""
DataLoader utilities and GPU PrefetchLoader.
"""

import threading
import queue

import torch
from torch.utils.data import DataLoader

from config import cfg
from data.dataset import MMVRDataset


class PrefetchLoader:
    """
    Wraps any DataLoader and prefetches the next batch onto the GPU
    while the current batch is being processed.

    This hides disk I/O and CPU preprocessing latency behind GPU compute,
    dramatically improving GPU utilisation on slow storage systems.

    Uses a single background thread — no multiprocessing, no deadlocks.

    Usage:
        loader = PrefetchLoader(DataLoader(...), device)
        for batch in loader:
            # batch tensors are already on GPU
            ...
    """

    def __init__(self, loader, device, queue_size=1):
        """
        Args:
            loader     : any PyTorch DataLoader
            device     : torch.device to prefetch tensors onto
            queue_size : number of batches to prefetch ahead (default 2)
        """
        self.loader     = loader
        self.device     = device
        self.queue_size = queue_size

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        # Queue holds prefetched batches; None signals end of data
        batch_queue = queue.Queue(maxsize=self.queue_size)

        def producer():
            """Background thread: loads batches and pushes to GPU."""
            try:
                for batch in self.loader:
                    # Move all tensors to GPU asynchronously
                    gpu_batch = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            # non_blocking=True overlaps H2D transfer with CPU work
                            gpu_batch[k] = v.to(self.device, non_blocking=True)
                        else:
                            gpu_batch[k] = v
                    batch_queue.put(gpu_batch)
            finally:
                batch_queue.put(None)   # sentinel

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            batch = batch_queue.get()
            if batch is None:
                break
            yield batch

        thread.join()


def wrap_with_prefetch(loader, device):
    """Wrap a DataLoader with PrefetchLoader for GPU prefetching."""
    return PrefetchLoader(loader, device)


def create_dataloaders_from_splits(train_samples, val_samples,
                                   test_samples, cfg):
    """Create DataLoaders from pre-split sample lists."""
    train_ds = MMVRDataset(train_samples, augment=True)
    val_ds   = MMVRDataset(val_samples,   augment=False)
    test_ds  = MMVRDataset(test_samples,  augment=False)

    loader_kwargs = dict(
        batch_size  = cfg.BATCH_SIZE,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"DataLoaders created:")
    print(f"  Train : {len(train_ds):,} samples  ({len(train_loader)} batches)")
    print(f"  Val   : {len(val_ds):,} samples  ({len(val_loader)} batches)")
    print(f"  Test  : {len(test_ds):,} samples  ({len(test_loader)} batches)")
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
