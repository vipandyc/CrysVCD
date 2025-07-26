import pathlib
import os

PARENT_PATH = pathlib.Path(__file__).parent.resolve()

def get_save_path(save_path, subfolder=""):
    # if save_path is not absolute, then it is relative to the parent path
    if not os.path.isabs(save_path):
        return str(PARENT_PATH / subfolder / save_path)
    else:
        return str(save_path)