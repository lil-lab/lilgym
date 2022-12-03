import json
from pathlib import Path
import os

# Path
data_path = str(Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()) + '/data/'

data_files = {
    "tower-scratch": {
        "train": "tower_scratch_train.json",
        "dev": "tower_scratch_dev.json",
        "test": "tower_scratch_test.json",
    },
    "tower-flipit": {
        "train": "tower_flipit_train.json",
        "dev": "tower_flipit_dev.json",
        "test": "tower_flipit_test.json",
    },
    "scatter-scratch": {
        "train": "scatter_scratch_train.json",
        "dev": "scatter_scratch_dev.json",
        "test": "scatter_scratch_test.json",
    },
    "scatter-flipit": {
        "train": "scatter_flipit_train.json",
        "dev": "scatter_flipit_dev.json",
        "test": "scatter_flipit_test.json",
    },
}


def get_data(appearance, starting_condition, split):
    env_name = f'{appearance}-{starting_condition}'
    if env_name not in data_files.keys():
        raise Exception("Data not found. Please check the spelling of appearance/starting_condition, or data file names/paths.")
    with open(os.path.join(data_path, data_files[env_name][split]), "r") as f:
        data = json.load(f)
    return data
