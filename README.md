<img alt="Supported by the Federal Ministry for Economic Affairs and Climate Action on the basis of a decision by the German Bundestag" src="./docs_src/source/BMWK.gif" height="250"/>

# Tangle Software

The unifying aim of the software published here is to facilitate applications of the mathematical theory of tangles.
Tangles can find and analyse structure in big data sets, support the development of expert systems in contexts 
such as medical diagnosis, help with the analysis of data in the social sciences and economics, and much more.

Applications of tangles, as well as the basics of tangle theory, are described in the book

    Reinhard Diestel
    Tangles: A structural approach to artificial intelligence in the empirical sciences
    Cambridge University Press 2024

The software published here implements many of the ideas and algorithms described in this book, 
and the two can be read side-by-side for maximum gain. However, the software has its own internal [documentation](https://tangle-software.github.io/tangles/) 
too, as well as numerous references to corresponding passages in the book.

Electronic access to the book will be via its dedicated [website](tangles-book.com).
This website will serve also as a low-threshold entry point to the software in this repository, 
by offering interactive examples of what tangles can do, 
how the software published here allows the user to interact with tangles it finds in real data, and so on.

# Getting started

To use this software library, a few dependencies are needed. We provide a conda environment file
that can be used to create an environment with all required dependencies to do tangle analysis.

To create this environmen you need to install the conda package manager,
for example via [miniconda](https://docs.conda.io/projects/miniconda/en/latest/). 
Once you've installed conda, open a conda prompt, 
navigate to the root of your repository checkout, and run:
```shell
conda env create -f environment.yml
conda activate tangles-dev
```

Make the `tangles` package avaiable within the conda environment:
```shell
conda develop tangles
```

Then check that your setup is ready for tangle analysis, by running our test-suite:
```shell
pytest -n auto
```

Now to get started with tangle analysis, we recommend to work through the tutorial notebooks in
`doc/source/tutorials`. As a next step, you should read the documentation of the library classes and 
functions that are used in those tutorials.  
