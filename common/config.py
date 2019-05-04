from pathlib import Path
import numpy as np

HOME_DIR = Path("~").expanduser()
PROJECT_DIR = Path(Path(__file__).absolute().parent.parent)
KERAS_DIR = HOME_DIR / ".keras"
MODELS_DATA_DIR = PROJECT_DIR / "models_data"

SERVER_URL = Path("http://199.247.6.121/")
# REMOTE_DATA_DIR = SERVER_URL / "data"

REMOTE_DATA_DIR = PROJECT_DIR / "data"
REMOTE_DATA_URL = Path(REMOTE_DATA_DIR.as_uri())

COLOR_PALETTE = np.array([[57, 106, 177], [218, 124, 48], [62, 150, 81], [204, 37, 41], [83, 81, 84],
                          [107, 76, 154], [146, 36, 40], [148, 139, 61]])
