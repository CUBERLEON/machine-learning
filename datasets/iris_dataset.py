from pathlib import Path

import pandas as pd

from common.config import REMOTE_DATA_URL, KERAS_DIR
from common.utils import download
from datasets.dataset import CompleteDataset


class IrisDataset(CompleteDataset):
    def __init__(self, remote_dir: Path = REMOTE_DATA_URL / "iris", cache_dir: Path = KERAS_DIR / "iris"):
        super().__init__(remote_dir=remote_dir, cache_dir=cache_dir)

    def _load_data(self):
        self._features_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
        self._target_col = 'iris class'

        self._data = pd.read_csv(download(self.remote_dir / "iris.data", self.cache_dir), names=self.features_cols + [self.target_col])
