# BIOBANK OLINK

### Setup

Work in the Conda environment:

```bash
conda create -n ethos python=3.10
```

Run this command to set up the repo in the development mode, to run experiments on a cluster `[all]`
is not required:

```bash
pip install -e .[all]
```

To obtain information about the capabilities of the package run the command below:

```bash
biobank_olink --help
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