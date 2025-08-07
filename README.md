# ASSESS
Estimating Node Abnormalities from Imprecise Subgraph-Level Supervision (Z. Peng, Y. Xue, Y. Wang, Q. Lin and C. Shen, TNSE 2025): [https://ieeexplore.ieee.org/document/11098617](https://ieeexplore.ieee.org/document/11098617) 

![image](https://github.com/zpeng27/ASSESS/blob/main/illustration.jpg)

## Overview
The repository is organized as follows:

- `data/` includes an example dataset and corresponding RWR random walk results;
- `models/` contains the implementation of the ASSESS pipeline (`weakad.py`);
- `layers/` contains the implementation of a standard GCN layer (`gcn.py`), the bilinear discriminator (`discriminator.py`), and the mean-pooling operator (`avgneighbor.py`);
- `utils/` contains the necessary processing tool (`process.py`).

You could further optimize the code based on your own needs. We display it in an easy-to-read form.

## Requirements

  * PyTorch 2.3.1
  * Python 3.11
  * NetworkX 3.3
  * graph-walker (https://github.com/kerighan/graph-walker)


## Usage

```python execute.py```

## Cite
Please cite our paper if you make use of ASSESS in your research:

```
@article{11098617,
title={Estimating Node Abnormalities From Imprecise Subgraph-Level Supervision},
author={Peng, Zhen and Xue, Yunqi and Wang, Yunfan and Lin, Qika and Shen, Chao},
journal={IEEE Transactions on Network Science and Engineering},
year={2025},
doi={10.1109/TNSE.2025.3593338}
}
```
