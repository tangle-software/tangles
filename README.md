<img alt="Supported by the Federal Ministry for Economic Affairs and Climate Action on the basis of a decision by the German Bundestag" src="./docs_src/source/BMWK.gif" height="250"/>

# Tangle Software

The unifying aim of the software published here is to facilitate applications of the mathematical theory of tangles.
Tangles can find and analyse structure in big data sets, support the development of expert systems in contexts 
such as medical diagnosis, help with the analysis of data in the social sciences and economics, and much more.

Applications of tangles, as well as the basics of tangle theory, are described in the _Tangles book_:

    Reinhard Diestel
    Tangles: A structural approach to artificial intelligence in the empirical sciences
    Cambridge University Press 2024

Electronic access to the book is available [here](https://www.tangles-book.com/book/), on the common [tangles website](https://www.tangles-book.com/) for all aspects of tangles, including the software offered here.

The software published here implements many of the ideas and algorithms described in this book, 
and the two can be read side-by-side for maximum gain. However, the software has its own internal [documentation](https://tangle-software.github.io/tangles/reference/api/tangles.html) 
too, as well as numerous references to corresponding passages in the book.

Our software collection also includes some interactive tutorials in the form of *Jupyter* notebooks, which allow you to familiarize yourself with how the various software modules can be used by working through some simple examples of tangle analysis. There are currently five such tutorials; see [here](https://www.tangles-book.com/software/) for an overview and some read-only trailers.


# Getting started

The tangle software is developed as a python library. In order to use it you need
- a copy of the tangle software source code
- a recent version of the python interpreter
- a number of other python packages that the tangle library depends on

In what follows we provide instructions for our recommended way to achieve this.

## Getting a copy of the tangle source code

If you are already familiar with the Git version control system, the easiest way is to clone this repository via
```shell
git clone https://github.com/tangle-software/tangles.git
```

As an alternative you can simply download a copy of this repository as a zip file, by clicking on the green "<> Code" button at the top, and choosing "Download ZIP". Then store and unpack the zip file
in a suitable location on your computer.


## Getting a recent version of the python interpreter and the project's dependencies

We follow the same approach as many projects in the scientific computing community, and recommend the use of the *conda package manager* to install python and the project dependencies.
The simplest way to install conda is with the help of the [miniconda package](https://docs.conda.io/projects/miniconda/en/latest/). You find installation instructions for all major operating systems on that same website.

Once you've installed conda, you need to open a *conda prompt*. On Windows you will find a new "Conda prompt" entry in your start menu, which opens a Windows command promp with approriate configuration. On MacOS and Linux you can simply start your favorite terminal (you might need to restart your terminal if the conda command is not recognized yet).

Now navigate to your copy of the tangle software library and run
```shell
conda env create -f environment.yml
```
This will create a new *conda environment* named `tangles-dev` with a suitable python interpreter version and all required dependencies. 
In order to work with this environment you need to activate it within your conda prompt. Simply execute the following:
```shell
conda activate tangles-dev
```

Now in order to use the tangle library within this newly created environment, you need to run the following command:

```shell
conda develop .
```

You can verify that your installation is working properly by executing our test-suite:
```shell
pytest -n auto
```

If everything works as expected, you should see a continuously updating status of the test execution. 
When test execution is done, you should see a summary line stating: "162 passed, 9 skipped, 1 warning".


## Explore our tangle tutorials

Congratulations, you're now ready to experiment with tangle analysis! An excellent starting point is our collection of tutorials,
provided as jupyter notebooks. The conda environment you've installed in a previous step contains an interactive environment to work with these notebooks.
In the conda prompt execute
```shell
jupyter notebook
```
which will start the jupyter notebook server. On most operating systems a browser tab should automatically open with the jupyter notebook interface, showing the directory
structure of the tangles library. In the folder *tutorials* you will find 5 interactive tutorials, showcasing different apllication areas of our tangle software. Double-clicking on a notebook file (ending in `.ipynb`) will open a new tab with the corresponding notebook. Enjoy exploring!