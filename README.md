# Enigma - An Efficient Model for Deciphering Regulatory Genomics

## Overview

This repository contains the Enigma model code, instructions for downloading model weights, and example usage code. 

## Setup

### Installation

```bash
conda env create -f environment.yml
conda activate enigma
pip install .
```

If you encounter any issues installing PyTorch and FlashAttention, please try commenting out `flash-attn` from `environment.yml` and installing FlashAttention separately (`pip install flash-attn==2.8.3`).

## Usage

We have provided pre-trained distilled model for non-commerical use (please see [License](#license)). 

Important: Enigma uses [FlashAttention](https://github.com/Dao-AILab/flash-attention) and currently only runs on GPUs that support FlashAttention. 

### Example Usage

Please check out [`examples/basic_setup.ipynb`](examples/basic_setup.ipynb)

More comprehensive usage examples, along with useful helper functions for inference and evaluations, will be 
added soon!

### Pre-trained Models

Pre-trained Enigma models are available as Wandb Artifacts:

- `deep-genomics-open-source/enigma/distilled-model:v0` ([link](https://wandb.ai/deep-genomics-open-source/enigma/artifacts/model/distilled-model/v0))

These models can be loaded using the following methods:
```python
from enigma.models import Enigma

# Requires Wandb login
model = Enigma.from_ckpt(wandb_artifact="deep-genomics-open-source/enigma/distilled-model:v0")

# Alternatively, manually download the checkpoint files from URLs
model = Enigma.from_ckpt(ckpt_path="path/to/downloaded/model.ckpt")
```

## Contact

Please contact andrew.jung (at) deepgenomics.com, andrewjung (at) psi.toronto.edu, or open a GitHub issue for questions. 

## Preprint and Citation

https://www.biorxiv.org/content/10.64898/2025.12.18.694875v1

```
@article {Jung2025.12.18.694875,
    author = {Jung, Andrew J and Zhu, Helen and Li, Roujia and Gao, Alice J. and Lau, Tammy T.Y. and Chu, Vivian S. and Lim, Declan and Cole, Christopher B. and Lee, Leo J. and Celaj, Albi and Frey, Brendan J.},
    title = {Enigma: An Efficient Model for Deciphering Regulatory Genomics},
    year = {2025},
    journal = {bioRxiv}
}
```

## Acknowledgements

We thank the authors of the following works we used for our project:

- The authors of [Borzoi](https://github.com/calico/borzoi) [1] for providing a comprehensive open-source repo to reproduce their dataset, model, training, and evaluations
- The authors of [gReLU](https://github.com/Genentech/gReLU) [2] and [Flashzoi](https://github.com/johahi/borzoi-pytorch/tree/main) [3] for sharing PyTorch versions of Borzoi, which we used for evaluations
- The authors of [AlphaGenome](https://github.com/google-deepmind/alphagenome) [4] for making the model available through their API, which we used for evaluations
- The authors of [Saluki](https://github.com/vagarwal87/saluki_paper) [5] and [RiboNN](https://github.com/Sanofi-Public/RiboNN) [6] for making their model and dataset available, which we used for downstream fine-tuning evaluations

If you use Enigma in your research, please also consider citing them where appropriate.

This work was also pursued as part of the first author's academic research at the University of Toronto. We would like to acknowledge Deep Genomics and the Vector Institute for compute resources.

## References

[1] Linder, J., Srivastava, D., Yuan, H., Agarwal, V., & Kelley, D. R. (2025). Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation. Nature Genetics, 57(4), 949-961.  
[2] Lal, A., Gunsalus, L., Nair, S., Biancalani, T., & Eraslan, G. (2025). gReLU: A comprehensive framework for DNA sequence modeling and design. Nature Methods, 1-5.  
[3] Hingerl, J. C., Karollus, A., & Gagneur, J. (2025). Flashzoi: an enhanced Borzoi for accelerated genomic analysis. Bioinformatics, 41(9), btaf467.  
[4] Avsec, Ž., Latysheva, N., Cheng, J., Novati, G., Taylor, K. R., Ward, T., ... & Kohli, P. (2025). AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model. bioRxiv, 2025-06.  
[5] Agarwal, V., & Kelley, D. R. (2022). The genetic and biochemical determinants of mRNA degradation rates in mammals. Genome biology, 23(1), 245.  
[6] Zheng, D., Persyn, L., Wang, J., Liu, Y., Ulloa-Montoya, F., Cenik, C., & Agarwal, V. (2025). Predicting the translation efficiency of messenger RNA in mammalian cells. Nature biotechnology, 1-14.

## License
This material is released under [Non-Commercial License](./LICENSE). If you use this software or data in your non-commercial work, please cite our paper.

For inquiries regarding commercial use please contact: legal@deepgenomics.com. 

(C) Deep Genomics Inc. (2025)
