# Monte Carlo with the repelled point processes (MCRPPy)

[![CI-tests](https://github.com/dhawat/MCRPPy/actions/workflows/ci.yml/badge.svg)](https://github.com/dhawat/MCRPPy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dhawat/MCRPPy/branch/main/graph/badge.svg?token=CODECOV_TOKEN)](https://codecov.io/gh/dhawat/MCRPPy)
[![Python >=3.8,<3.10](https://img.shields.io/badge/python->=3.8,<3.10-blue.svg)](https://www.python.org/downloads/release/python-371/)

> Sampling repelled point processes, estimating function integrals using various Monte Carlo methods (including a method with the repelled point process), and illustrating 2D gravitational allocation from the Lebesgue measure to a point process.

- [Monte Carlo with the repelled point processes (MCRPPy)](#monte-carlo-with-the-repelled-point-processes-mcrppy)
  - [Introduction](#introduction)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
    - [Install the project as a dependency](#install-the-project-as-a-dependency)
    - [Install in editable mode and potentially contribute to the project](#install-in-editable-mode-and-potentially-contribute-to-the-project)
  - [Getting started](#getting-started)
    - [Companion paper](#companion-paper)
    - [Notebooks](#notebooks)
  - [How to cite this work](#how-to-cite-this-work)

## Introduction

`MCRPPy` is an open-source Python package that currently includes methods for sampling from a variety of point processes, including the homogeneous Poisson, Thomas, Ginibre, scrambled Sobol, Binomial, and their repelled counterparts. The project also includes several Monte Carlo methods, including a Monte Carlo method with the repelled point process.
Furthermore, the package provides tools for visualizing the gravitational allocation from the Lebesgue measure to a point process within a two-dimensional space (d=2).

This project serves as a companion code for the research paper titled [''Repelled point processes with application to numerical integration''](TBC); [see: How to cite this work](#how-to-cite-this-work).

## Dependencies

- `MCRPPy` works with [![Python >=3.8,<3.10](https://img.shields.io/badge/python->=3.8,<3.10-blue.svg)](https://www.python.org/downloads/release/python-371/).

- Python dependencies are listed in the [`pyproject.toml`](./pyproject.toml) file.

## Installation

### Install the project as a dependency

- Install from source (this may be broken)

  ```bash
  # activate your virtual environment and run
  poetry add git+https://github.com/dhawat/MCRPPy.git
  # pip install git+https://github.com/dhawat/MCRPPy.git
  ```

### Install in editable mode and potentially contribute to the project

The package can be installed in **editable** mode using [`poetry`](https://python-poetry.org/).

To do this, clone the repository:

- if you considered [forking the repository](https://github.com/dhawat/MCRPPy/fork)

  ```bash
  git clone https://github.com/your_user_name/MCRPPy.git
  ```

- if you have **not** forked the repository

  ```bash
  git clone https://github.com/dhawat/MCRPPy.git
  ```

and install the package in editable mode

```bash
cd mcrppy
poetry shell  # to create/activate local .venv (see poetry.toml)
poetry install
# poetry install --no-dev  # to avoid installing the development dependencies
# poetry add -E docs -E MCRPPy  # to install extra dependencies
```

## Getting started

### Companion paper

We wrote a companion paper to `MCRPPy`, "[Repelled point processes with application to numerical integration](TBC)".

In the paper, we introduce the repelled point process, analyze its properties, and use it to develop a Monte Carlo estimator for approximating function integrals.
Our main theoretical result is that the repelled Poisson point process yields an unbiased Monte Carlo estimator with lower variance than the crude Monte Carlo method.
On the computational side, the evaluation of our estimator is only quadratic in the number of integrand evaluations and can be easily parallelized without any communication across tasks.
We illustrate the variance reduction result with numerical experiments and compare it to popular Monte Carlo methods.
Finally, we numerically investigate a few open questions on the repulsion operator.

### Notebooks

We provide three tutorial Jupyter Notebooks available in the [./notebooks](./notebooks) folder.

- ``tutorial_sample_repelled_point_pattern.ipynb``: tutorial for sampling a Repelled point process.
- ``tutorial_monte_carlo_methods.ipynb``: tutorial for estimating function integrals using the available Monte Carlo methods.
- ``tutorial_gravitational_allocation.ipynb``: tutorial for illustrating a two-dimensional gravitational allocation from Lebesgue to a point process.

We also provide two Jupyter Notebooks for replicating the study of the [companion paper](#companion-paper).

- ``companion_paper.ipynb``: main notebook.
- ``structure_factor_and_pcf.ipynb``: supplementary notebook.

See the README.md in the [./notebooks](./notebooks) folder for further instructions on how to run a notebook locally.

## How to cite this work

If `MCRPPy` has been significant in your research, and you would like to acknowledge the project in your academic publication, please consider citing it with this piece of BibTeX:

  ```bash
  @article{HBLR2023,
    arxivid = {TBC},
    journal = {TBC},
    author  = {Hawat, Diala and Bardenet, R{\'{e}}mi and Lachi{\`{e}}ze-Rey, Rapha{\"{e}}l},
    title   = {Repelled point processes with application to numerical integration},
    year    = {2023},
  }
  ```
