from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR_STR = str(ROOT_DIR)

if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)
