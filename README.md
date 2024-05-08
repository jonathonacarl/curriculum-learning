# Curriculum Learning

A Deep Learning (CSCI 381) final project by Jon Carl and Nate Lentz on Curriculum Learning.

## Files to view:

* `curriculum-learning.ipynb`: **This is the main file to review**. The jupyter notebook containing our review of the Curriculum Learning paper by Bengio et al. and the curriculum learning experiments. The notebook imports methods developed in the three `.py` files mentioned below.

* `model.py`: Contains
  - The FeedForward neural network used for curriculum learning.
  - Training and evaluation methods for our models.
  - Grid Search across learning rate and hidden size hyperparameters.
  - Early stopping logic for training models.

* `datagen.py`: Methods include
  - Generating shapes data for the curriculum experiments by Bengio et al.
  - Setting up shape data into torch datasets/loaders.
  - Generating data loaders with a proportion $p$ of complex data and $1-p$ of basic data, $0 \leq p \leq 1$. Used for **gradual switching** experiment.
  
* `experiment.py`: Methods for curriculum experiments (with and without **gradual switching**).

## Quick Recap of other notable items in the repo (not necessary to view):

* `models`: contains the models trained from `curriculum-learning.ipynb`.
* `figures`: contains the training and validation plots from each of the models trained from `curriculum-learning.ipynb`.
