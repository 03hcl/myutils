from typing import Any, Iterator, List, Tuple

from torch.utils.data import BatchSampler, DataLoader


class PreSampledBatchSampler(BatchSampler):

    def __init__(self, sampler, batch_size, drop_last):
        super(PreSampledBatchSampler, self).__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.pre_sampled: Tuple[List[Any], ...] = tuple(_iterate_batch(sampler, batch_size, drop_last))

    def __iter__(self):
        for batch in self.pre_sampled:
            yield batch


def _iterate_batch(sampler, batch_size, drop_last) -> Iterator[List[Any]]:
    batch: List[Any] = []
    for sample in sampler:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0 and not drop_last:
        yield batch


class PreSampledDataLoader(DataLoader):

    __initialized_subclass = False

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 # batch_sampler=None,
                 num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None,
                 # multiprocessing_context=None,
                 ):
        super(PreSampledDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            # batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn,
            # multiprocessing_context=multiprocessing_context
        )
        if batch_size is not None:
            # noinspection PyUnresolvedReferences
            self.batch_sampler = PreSampledBatchSampler(self.sampler, batch_size, drop_last)
        self.__initialized_subclass = True

    def __setattr__(self, attr, val):
        if self.__initialized_subclass and attr in ("batch_size", "sampler", "drop_last"):
            raise ValueError("{} attribute should not be set after {} is "
                             "initialized".format(attr, self.__class__.__name__))
        super(DataLoader, self).__setattr__(attr, val)
