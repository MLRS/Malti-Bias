## Dropout Regularization
Based on [this paper](https://research.google/pubs/measuring-and-reducing-gendered-correlations-in-pre-trained-models/).

BERT includes two key dropout parameters that can be configured:

1. **Attention Dropout (`a`)**: Applied to the attention weights.
2. **Hidden Activation Dropout (`h`)**: Applied to the hidden layer activations.

By default, both parameters are set to `0.1`. In our experiments, we tested these dropout rates with values of `0.1`, `0.15`, and `0.2` to observe their effects on model performance.

## Installation

To get started, make sure you have the necessary package installed. Run the following command to install the `datasets` package:

```bash
pip install datasets

