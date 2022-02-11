from logging import root
import os
from pathlib import Path
import sys

root_path = Path("")
root_str = root_path.absolute()

sys.path.append(root_str)


def get_project_root() -> Path:
    root = os.path.dirname(os.path.abspath(__file__))
    return Path(root)
