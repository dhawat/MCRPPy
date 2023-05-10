# Monte Carlo with the repelled point processes (MCRPPy)

[![CI-tests](https://github.com/dhawat/MCRPPy/actions/workflows/ci.yml/badge.svg)](https://github.com/dhawat/MCRPPy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dhawat/MCRPPy/branch/main/graph/badge.svg?token=CODE_COV_TOKEN)](https://codecov.io/gh/dhawat/MCRPPy)
[![Python >=3.8,<3.10](https://img.shields.io/badge/python->=3.8,<3.10-blue.svg)](https://www.python.org/downloads/release/python-371/)

> Sample a repelled point process and compute a Monte Carlo estimation for the integral of a function using various variants of the Monte Carlo method, including the Monte Carlo with a repelled point process.

- [Monte Carlo with the repelled point processes (MCRPPy)](#monte-carlo-with-the-repelled-point-processes-mcrppy)
  - [Introduction](#introduction)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
    - [Install the project as a dependency](#install-the-project-as-a-dependency)
    - [Install in editable mode and potentially contribute to the project](#install-in-editable-mode-and-potentially-contribute-to-the-project)
  - [How to cite this work](#how-to-cite-this-work)
    - [Companion paper](#companion-paper)
    - [Citation](#citation)

## Introduction

`mcrppy` is an open-source Python project that currently includes methods for sampling from a variety of point processes, including the homogeneous Poisson, Thomas, Ginibre, Scrambled Sobol, Binomial, and their repelled counterparts. The project also includes several variants of the Monte Carlo method, including Monte Carlo with a repelled point process.
This project serves as a companion code for the research paper titled ``Monte Carlo with the repelled Poisson point process``; [see: How to cite this work](#how-to-cite-this-work).

## Dependencies

- `mcrppy` works with [![Python >=3.8,<3.10](https://img.shields.io/badge/python->=3.7.1,<3.10-blue.svg)](https://www.python.org/downloads/release/python-371/).

- Python dependencies are listed in the [`pyproject.toml`](./pyproject.toml) file.

## Installation

### Install the project as a dependency

- Install from source (this may be broken)

  ```bash
  # activate your virtual environment and run
  poetry add git+https://github.com/dhawat/MCRPPy.git
  # pip install git+https://github.com/For-a-few-DPPs-more/MCRPPy.git
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
# poetry add -E docs -E structure-factor  # to install extra dependencies
```

## How to cite this work

### Companion paper

We wrote a companion paper to `mcrppy`, "[Monte Carlo with the repelled Poisson point process](TBC)". In the paper, we introduced the repelled point process, analyzed its properties, and utilized it to develop a variance reduction Monte Carlo method (MCR). We also conducted a comparison study of MCR against other competing Monte Carlo methods.

### Citation

If `mcrppy` has been significant in your research, and you would like to acknowledge the project in your academic publication, please consider citing it with this piece of BibTeX::

  ```bash
  @article{HBLR2023,
    arxivid = {TBC},
    journal = {TBC},
    author  = {Hawat, Diala and Bardenet, R{\'{e}}mi and Lachi{\`{e}}ze-Rey, Rapha{\"{e}}l},
    note    = {TBC},
    title   = {Monte Carlo with the repelled Poisson point process},
    year    = {2023},
  }
  ```
