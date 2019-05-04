from pathlib import Path

from pandas import DataFrame

from common.config import REMOTE_DATA_URL, KERAS_DIR
from common.utils import download
from datasets.dataset import CompleteDataset


class MushroomsDataset(CompleteDataset):
    def __init__(self, remote_dir: Path = REMOTE_DATA_URL / "mushrooms", cache_dir: Path = KERAS_DIR / "mushrooms"):
        super().__init__(remote_dir=remote_dir, cache_dir=cache_dir)

    def _load_data(self):
        maps = [{'e': 1, 'p': 0},
                {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 'k': 'knobbed', 's': 'sunken'},
                {'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'},
                {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'r': 'green', 'p': 'pink',
                 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
                {'t': 1, 'f': 0},
                {'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy', 'f': 'foul', 'm': 'musty',
                 'n': 'none', 'p': 'pungent', 's': 'spicy'},
                {'a': 'attached', 'd': 'descending', 'f': 'free', 'n': 'notched'},
                {'c': 'close', 'w': 'crowded', 'd': 'distant'},
                {'b': 'broad', 'n': 'narrow'},
                {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray', 'r': 'green',
                 'o': 'orange', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
                {'e': 'enlarging', 't': 'tapering'},
                {'b': 'bulbous', 'c': 'club', 'u': 'cup', 'e': 'equal',
                 'z': 'rhizomorphs', 'r': 'rooted', '?': 'missing'},
                {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
                {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
                {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink',
                 'e': 'red', 'w': 'white', 'y': 'yellow'},
                {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink',
                 'e': 'red', 'w': 'white', 'y': 'yellow'},
                {'p': 'partial', 'u': 'universal'},
                {'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'},
                {'n': 'none', 'o': 'one', 't': 'two'},
                {'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring', 'l': 'large', 'n': 'none',
                 'p': 'pendant', 's': 'sheathing', 'z': 'zone'},
                {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'r': 'green',
                 'o': 'orange', 'u': 'purple', 'w': 'white', 'y': 'yellow'},
                {'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered', 'v': 'several', 'y': 'solitary'},
                {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'u': 'urban', 'w': 'waste', 'd': 'woods'}]

        mushrooms = []
        with open(download(self.remote_dir / "agaricus-lepiota.data", self.cache_dir)) as f:
            for line in f.readlines():
                tokens = line.strip().split(',')

                mushrooms.append([map[token] for token, map in zip(tokens, maps)])

        self._features_cols = ["cap_shape", "cap_surface", "cap_color",
                               "has_bruises",
                               "odor",
                               "gill_attachment", "gill_spacing", "gill_size", "gill_color",
                               "stalk_shape", "stalk_root",
                               "stalk_surface_above_ring", "stalk_surface_below_ring",
                               "stalk_color_above_ring", "stalk_color_below_ring",
                               "veil_type", "veil_color",
                               "ring_number", "ring_type",
                               "spore_print_color",
                               "population", "habitat"]
        self._target_col = "is_eatable"

        self._data = DataFrame(data=mushrooms,
                               columns=self.features_cols + [self.target_col])
