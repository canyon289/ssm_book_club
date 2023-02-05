# State Space Model Book Club

## Resources
* **Community** - https://community.intuitivebayes.com/
* **Kick off Blog Post** - https://ravinkumar.com/ssm-book-club.html
* **Livestreams** - https://www.youtube.com/@ravink/streams
* **ProbML Book** - https://probml.github.io/pml-book/book2.html
  * Chapter 8
  * Chapter 9
  * Chapter 29

## Environment Setup
The best way to learn is to get hands on with the code. 
Installing an environment gives you the most control.

There are many python environment management tools, learn whatever tool you want, 
we'll be using conda by default.

### Create a Workspace

Clone this repo
```bash
git clone git@github.com:canyon289/ssm_book_club.git
```

Move into the repo:
```bash
cd ssm_book_club
```

### Create conda environment (not for Windows users - see below)

#### Unix/Mac Users

1. Create the Conda Environment:
```bash
conda env create -f environment.yml
```

2. Activate the new environment:
```bash
conda activate ssm_book_club
```

#### Windows Users

The installation of the `dynamax` dependency `jaxlib` runs some issues on Windows.
One possible workaround is the following:

1. Remove the pip dependencies at the bottom of the `environment.yml` file.
```bash
- pip:
    - dynamax @ git+https://github.com/probml/dynamax
```

2. Create the conda environment using:
```bash
conda env create -f environment.yml
```

3. Activate the new environment
```bash
conda activate ssm_book_club
```

4. Pip install jaxlib with, for example:
```bash
pip install "jax[cpu]===0.3.25" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
```
Note that this will install the CPU only version of jax (and jaxlib). 
If you are interested in adding GPU support see additional information here `https://github.com/cloudhan/jax-windows-builder#unstable-builds`.
Version `0.3.25` is the latest available as of 2023-01-21. 
For more versions see the list at `https://whls.blob.core.windows.net/unstable/index.html` and release information at `https://github.com/google/jax/releases`.

5. Pip install dynamax with:
```bash
pip install dynamax[notebooks]
```
