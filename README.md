# ASSESS
Estimating Node Abnormalities from Imprecise Subgraph-Level Supervision

![illustration](https://github.com/user-attachments/assets/eba3efe7-6cee-4683-9943-abca0c8058c6)


## Overview
The repository is organized as follows:

- `data/` includes an example dataset and corresponding RWR random walk results;
- `models/` contains the implementation of the ASSESS pipeline (`weakad.py`);
- `layers/` contains the implementation of a standard GCN layer (`gcn.py`), the bilinear discriminator (`discriminator.py`), and the mean-pooling operator (`avgneighbor.py`);
- `utils/` contains the necessary processing tool (`process.py`).

We are willing to release a more comprehensive version including ASSESS++ after the review process.

## Requirements

  * PyTorch 2.3.1
  * Python 3.11
  * NetworkX 3.3
  * graph-walker (https://github.com/kerighan/graph-walker)


## Usage

```python execute.py```
