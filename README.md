
# NAM_Entropy

> Reproducible research code for computing and analyzing entropy-based metrics and related experiments.
> Built with **Python 3.12**, **Poetry**, and **Jupyter Notebooks** for a clean, portable workflow.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](#)
[![Poetry](https://img.shields.io/badge/packaging-poetry-60A5FA.svg)](#)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](#license)


## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Soft Entropy Estimates](#soft-entropy-estimates)
- [References](#references)


## Overview

**NAM_Entropy** provides utilities and experiments for entropy-related analysis. Typical use cases include:

- Computing **Shannon entropy**, **cross-entropy**, **KL-divergence**, and related information-theoretic measures.
- Understanding the efficacy of sentence-level and token-level embeddings in a model by computing  its **information**, **variation**, **regularity**, **disentanglement**.


Our target audience is:

- Cognitive Scientists who want an easy way to understand information-theoretic metrics for the information that a Nerual Network model carry about token and sentences via their embeddings.
- Maching Learning researchers who are interested in using parts of our efficient implementation in their workflow to understand the entropy and mutual information in their latent space model representations.
- Other researchers interested in computing entropy and information theory to their work.



## Installation

To install the package dependencies, you can use either [poetry](https://python-poetry.org/) or [conda](https://anaconda.org/anaconda/conda) after changing to the package directory (containing `pyproject.toml` and `environment.yml`).   

### Install with Poetry
```
poetry install
```
(This creates a poetry virtual environment: `nam-entropy-<hash>`.)

### Install with Conda  
```
conda env create -f environment.yml
```
(This creates the conda virtual environment: `nam-entropy-conda`.)





## Project Structure

```text
nam_entropy/
├─ pyproject.toml
├─ poetry.lock
├─ README.md
├─ LICENSE
├─ .gitignore
├─ Makefile                           # optional convenience commands
├─ .env.example                       # example environment variables
├─ notebooks/
│  ├─ A. Automatic entropy and information calculations.ipynb
│  ├─ B. Interactive entropy and information calculations.ipynb
│  └─ C. Programmatic entropy and information calculations.ipynb
├─ src/
│  └─ nam_entropy/
│     ├─ __init__.py
│     ├─ make_data.py
│     ├─ data_prep.py
│     ├─ bin_distribution_plots.py
│     ├─ integrated_distribution_2d_sampler.py
│     └─ h.py
└─ tests/
```






## Soft Entropy Estimates

This package uses the soft entropy estimation methodology in **[Conklin (2025), Section 5.4](#primary-theoretical-framework)** 
to provide a discrete approximation to differentiable entropy computation. This approach enables
gradient-based optimization while maintaining the interpretability of traditional entropy
measures. The key insight is mapping continuous neural representations to probability
distributions over learned bin centers, preserving both representational nuances and
computational tractability.


   
### Metrics

#### 1. Entropy
We compute the __entropy__ $H(Z)$ of our given vector-valued distribution $Z$ by creating 
an associated soft-binned probability distrubtion $\mathrm{SoftBin}(Z)$ on a finite set of points 
randomly associated to the given dataset, 
and then compute the Shannon entropy $H(\mathrm{SoftBin}(Z))$ of this finite distribution.


For more details on Shannon entropy, see 
[Wikipedia: Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))
or 
[Science Direct: Shannon Entropy](https://www.sciencedirect.com/topics/computer-science/shannon-entropy).




#### 2. Conditional Entropy
We also can compute the conditional entropy of the given vector-valued distribution $Z$ 
with respect to a given discrete labelling of $Z$.  We think of this as a (categorical) label random variable $L$
valued in the finite set of labels that shares a common probability space with $Z$.  Then we compute 
the __conditional entropy__ $H(Z|L)$ of $Z$ conditioned on (knowing) $L$ by the usual formula

$$H(Z|L) = \sum_{l \in L} p(l) H(Z|L=l)$$

where the entropy conditioned on the particular label value $L = l$ is given by 

$$H(Z|L=l) = -\sum_{z \in Z} p(z|l) \log p(z|l),$$

and where
- $Z$ := Population data distribution (continuous vector-valued RV),
- $L$ := Sub-population label (categorical RV).


For more details on condtional entropy, see 
[Wikipedia: Conditional Entropy](https://en.wikipedia.org/wiki/Conditional_entropy) 
or 
[Science Direct: Conditional Entropy](https://www.sciencedirect.com/topics/computer-science/conditional-entropy).




#### 3. Mutual Information
The mutual information describes the amount of informatino that of two random variables give about each other.
By definition, the __mutual information__ $I(Z, L)$ of the two random variables $Z$ and $L$ is given by

$$I(Z, L) := H(Z) - H(Z|L),$$

where 
- $Z$ := Population data distribution (continuous vector-valued RV),
- $L$ := Sub-population label (categorical RV).


For more details on mutual information, see 
[Wikipedia: Mutual Information](https://en.wikipedia.org/wiki/Mutual_information) 
or 
[Science Direct: Mutual Information](https://www.sciencedirect.com/topics/computer-science/mutual-information).







  ## References

  ### Primary Theoretical Framework

  **Conklin, H.** (2025). *Information Structure in Mappings: An Approach to Learning,
  Representation, and Generalisation.* PhD Thesis, University of Edinburgh. arXiv:2505.23960.
  [https://arxiv.org/abs/2505.23960](https://arxiv.org/abs/2505.23960)

  For the mathematical foundations of **soft entropy estimation** used in this project, see
  **Section 5.4: "Soft Entropy Estimation"** (pp. 94-99), which provides the theoretical basis 
  for the continuous-to-discrete entropy approximation methods implemented in our codebase.


  ### Related Publications

  **Conklin, H., & Smith, K.** (2024). Representations as Language: An Information-Theoretic
  Framework for Interpretability. *International Meeting of the Cognitive Science Society*.
  arXiv:2406.02449. [https://arxiv.org/abs/2406.02449](https://arxiv.org/abs/2406.02449)

  **Conklin, H., & Smith, K.** (2024). Compositionality With Variation Reliably Emerges in
  Neural Networks. *International Conference on Learning Representations (ICLR)*.
  [https://openreview.net/forum?id=-Yzz6vlX7V-](https://openreview.net/forum?id=-Yzz6vlX7V-)



