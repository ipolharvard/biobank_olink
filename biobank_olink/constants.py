from importlib.resources import files

PROJECT_DATA = files("biobank_olink.data")
PROJECT_ROOT = PROJECT_DATA / ".." / ".."

SEED = 42
