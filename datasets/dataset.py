from abc import ABC
from pathlib import Path

from pandas import DataFrame


class Dataset(ABC):
    def __init__(self, remote_dir: Path, cache_dir: Path):
        self.__remote_dir = remote_dir
        self.__cache_dir = cache_dir
        print(f'Initializing dataset "{self.remote_dir}" and caching it in "{self.cache_dir}"')

    def _load_data(self):
        raise NotImplementedError

    @property
    def remote_dir(self):
        return self.__remote_dir

    @property
    def cache_dir(self):
        return self.__cache_dir

    def __len__(self):
        raise NotImplementedError


class IndexedDataset(Dataset, ABC):
    def __init__(self, remote_dir: Path, cache_dir: Path):
        super().__init__(remote_dir=remote_dir, cache_dir=cache_dir)
        self._indices = None
        self._load_data()
        assert self._indices is not None

    def sample(self, index: int):
        raise NotImplementedError

    @property
    def indices(self):
        return self._indices

    def __len__(self):
        return len(self._indices)


class CompleteDataset(Dataset, ABC):
    def __init__(self, remote_dir: Path, cache_dir: Path):
        super().__init__(remote_dir=remote_dir, cache_dir=cache_dir)
        self._data = None
        self._features_cols = None
        self._target_col = None
        self._load_data()
        assert self._data is not None and self._features_cols is not None and self._target_col is not None

    @property
    def data(self) -> DataFrame:
        return self._data

    @property
    def features_cols(self):
        return self._features_cols

    @property
    def target_col(self):
        return self._target_col
