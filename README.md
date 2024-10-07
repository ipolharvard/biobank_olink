# BIOBANK OLINK

### Setup

Work in the Conda environment:

```bash
conda create -n olink python=3.12
```

Run this command to set up the repo in the development mode. To run experiments related to
clustering
add this extension`[cluster]`:

```bash
pip install -e .
```

### Data

Biobank data is supposed to be in the data directory:

```
.
├── pyproject.toml
├── README.md
├── scripts
└── biobank_olink
    └── data  <--- here
```