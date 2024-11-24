# Coding Challenge Overview

To complete this test we provide a `main.py` file that is prepopulated to load data, train a network, and calculate network bounds. There are places in the code that need completing for it to run correctly.

## Setup Requirements

1. Ensure Poetry is installed on your system. Refer to the [Poetry installation guide](https://python-poetry.org/docs/#installation) if needed.
   
2. Activate the virtual environment managed by Poetry by running:
   
```bash
poetry shell
```

3. Install dependencies:

```bash
poetry install
```

4. Generate data:

```bash
poetry run python -m provided.data --verbose
```

5. Run the code:

```bash
poetry run python ./main.py
# Running main.py...
```

If using VSCode, select the Python interpreter via the command palette (`Ctrl+Shift+P`) and choose the virtual environment created by Poetry.

---

## Task 1. Loading Data with Multi-Processing

Resources: `./provided/data.py`, `./provided/loader.py`

The first stage involves loading data from a CSV file in preparation for training. The data consists of a linearly separable dataset with 3 features (x, y, and z) and 2 classes {0, 1}. The helper function generates this dataset and saves it to `./provided/3d_data.csv`.

### Objective

Implement multi-processing to load the data across parallel threads. Note this will not be faster than a single-threaded implementation but rather a way of demonstrating knowledge of multi-processing in Python.

### Instructions:

1. Complete the `MultiProcessDataset` class in `./provided/loader.py`.

---

## Task 2. Neural Network Implementation

Resources: `./provided/network.py`

The second stage involves training a simple feed-forward neural network to compute gradients and update weights. The network consists of an input layer, two hidden layers (4 nodes each), and an output layer (single node) of one dimension. Note there are no activation functions.

### Objective

Implement the backward pass to update the weights and biases of the neural network using the activations computed during the forward pass.

### Instructions:

1. Implement the backward pass method `SimpleNeuralNetwork.backward()`, using the mean squared error as the loss function.

---

## Task 3. Evaluating the Neural Network Output Bounds

Resources: `./provided/bounds.py`

The final stage involves evaluating the output bounds of the neural network, this is known as Interval Bound Propagation (IBP). The goal is to compute the lower and upper bounds of the output of the network given the input bounds. This task helps in understanding the behavior of the network under uncertain or perturbed inputs.

Given an input X with known lower and upper bounds, `X_lower` and `X_upper`, we aim to compute how these bounds propagate through a linear layer of the network defined by weights `W` and bias `b`. We need to calculate the new output bounds after this layer: `output_lower` and `output_upper`.

### Objective

Implement a method to compute the output bounds of a single linear layer given the input bounds. 

### Instructions:

1. Complete the network training from section 2.

2. Implement the `propagate_bounds` method of the `IntervalBoundPropagation` class. This performs the linear operations of that layer on the input bounds. 

3. Run the code from `main.py` to evaluate the output bounds of the network.