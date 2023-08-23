from importlib.resources import files

DATA_DIR = files("biobank_olink.data")
PROJECT_ROOT = DATA_DIR / ".." / ".."
