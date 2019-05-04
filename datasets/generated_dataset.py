from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
import numpy as np

from datasets.dataset import CompleteDataset


class GeneratedDataset(CompleteDataset):
    def __init__(self, n_samples=1000, n_features=9, n_classes=2, n_informative=4, n_redundant=4,
                 n_clusters_per_class=1, class_sep=1., random_state=None,
                 remote_dir: Path = None, cache_dir: Path = None):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_clusters_per_class = n_clusters_per_class
        self.class_sep = class_sep
        self.random_state = random_state
        super().__init__(remote_dir=remote_dir, cache_dir=cache_dir)

    def _load_data(self):
        x, y = make_classification(n_samples=self.n_samples, n_features=self.n_features, n_classes=self.n_classes,
                                   n_informative=self.n_informative, n_redundant=self.n_redundant,
                                   n_clusters_per_class=self.n_clusters_per_class, class_sep=self.class_sep,
                                   random_state=self.random_state)

        self._features_cols = [f"feature {i}" for i in range(self.n_features)]
        self._target_col = "class"

        self._data = pd.DataFrame(data=np.hstack([x, y.reshape((self.n_samples, 1))]), columns=self.features_cols + [self.target_col])
