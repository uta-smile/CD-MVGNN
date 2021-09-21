from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dglt.contrib.moses.moses.utils import set_torch_seed_to_all_gens
from dglt.multi_gpu_wrapper import MultiGpuWrapper as mgw


class MosesTrainer(ABC):
    @property
    def n_workers(self):
        n_workers = self.config.n_workers
        return n_workers if n_workers is not None else 0

    def get_collate_device(self, model):
        n_workers = self.n_workers
        return 'cpu' if n_workers > 0 else model.device

    def get_dataloader(self, model, data, collate_fn=None, shuffle=True, epoch=1):
        if collate_fn is None:
            collate_fn = self.get_collate_fn(model)
        sampler = None
        if self.config.enable_multi_gpu:
            sampler = DistributedSampler(data,
                                         num_replicas=mgw.size(),
                                         rank=mgw.rank())
            sampler.set_epoch(epoch)
            shuffle = False
        return DataLoader(data, batch_size=self.config.n_batch, sampler=sampler,
                          shuffle=shuffle, pin_memory=self.get_collate_device(model) == 'cpu',
                          num_workers=self.n_workers, collate_fn=collate_fn,
                          worker_init_fn=set_torch_seed_to_all_gens
                          if self.n_workers > 0 else None)

    def get_collate_fn(self, model):
        return None

    @abstractmethod
    def get_vocabulary(self, data, extra=None):
        pass

    @abstractmethod
    def fit(self, model, train_data, val_data=None):
        pass
