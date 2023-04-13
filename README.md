# Protein-Protein-Interaction
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](<https://opensource.org/licenses/MIT>)

PPI prediction on HuRi data using contrastive learning.

## Installation

These instructions assume a working installation of [Anaconda](https://www.anaconda.com/).

```bash
git clone git@github.com:shreayan98c/Protein-Protein-Interaction.git
cd Protein-Protein-Interaction
conda env create -f environment.yml
```

Depending on your desired configuration, you may need to install the
[PyTorch](https://pytorch.org/get-started/locally/) package separately. This can be done following
the instructions on the PyTorch website, in an empty conda environment. Then you can install the
remaining packages with:

```bash
conda activate PPI_Pred
pip install -r requirements.txt
pip install -e .
```

This is only necessary if the installation from `environment.yml` fails.

## Usage

```bash
python main.py train
```

## License

This project is licensed under the terms of the MIT license.