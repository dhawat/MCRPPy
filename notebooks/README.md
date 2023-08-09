# Notebooks

We provide three tutorial Jupyter Notebooks.

- ``tutorial_sample_repelled_point_pattern.ipynb``: tutorial for sampling a Repelled point process.
- ``tutorial_monte_carlo_methods.ipynb``: tutorial for estimating function integrals using the available Monte Carlo methods.
- ``tutorial_gravitational_allocation.ipynb``: tutorial for illustrating a two-dimensional gravitational allocation from Lebesgue to a point process.

We also provide two Jupyter Notebooks for replicating the study of the companion paper
[''Repelled point processes with application to numerical integration''](TBC).

- ``companion_paper.ipynb``: main notebook.
- ``structure_factor_and_pcf.ipynb``: supplementary notebook.

<!-- ## Run a notebook remotely

- `example.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guilgautier/template-python-project/blob/main/notebooks/example.ipynb) -->

## Run a notebook locally

1. It is suggested you clone the repository and get the latest version of the source code

    ```bash
    git clone https://github.com/dhawat/MCRPPy.git
    cd MCRPPy
    git pull origin main
    ```

2. Then, install the project in a virtual environment, see also the [installation instructions on GitHub](https://github.com/For-a-few-DPPs-more/spatstat-interface/blob/main/README.md#Installation)

    - if you use [`poetry`](https://python-poetry.org/)

        ```bash
        # cd template-python-project
        poetry shell  # to create/activate local .venv (see poetry.toml)
        poetry install -E notebook  # (see [tool.poetry.extras] in pyproject.toml)
        ```

    - if you use [`pip`](https://pip.pypa.io/en/stable/)

        ```bash
        # cd template-python-project
        # activate a virtual environment of your choice and run
        pip install '.[notebook]'  # (see [tool.poetry.extras] in pyproject.toml)
        ```

3. Finally, launch the notebook

    - with your favorite notebook interface, VSCode for example
    - or if you use [`poetry`](https://python-poetry.org/), run

        ```bash
        # cd template-python-project
        poetry run jupyter notebook
        ```

    - or if you don't use poetry, run

        ```bash
        # cd template-python-project
        jupyter notebook
        ```
