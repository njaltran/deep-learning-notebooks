# Deep Learning Marimo Notebooks

Interactive `marimo` notebooks for learning core deep learning concepts, sequence models, transformers, and modern LLM techniques.

## Contents

- `01_neurons_and_networks.py` — perceptrons, activations, MLPs, and decision boundaries
- `02_training_deep_networks.py` — loss functions, gradient descent, backpropagation, and overfitting
- `03_convolutional_networks.py` — convolutions, filters, CNN architectures, and image classification
- `04_sequence_models.py` — RNNs, vanishing gradients, LSTMs, and character generation
- `05_attention_mechanism.py` — scaled dot-product attention and multi-head attention
- `06_transformer_architecture.py` — positional encoding, transformer blocks, masking, and a toy transformer
- `07_language_models.py` — tokenization, causal masking, sampling, and a mini GPT
- `08_modern_llm_techniques.py` — LoRA, embeddings, RAG, and RLHF concepts

## Quickstart

### With `uv`

```bash
uv sync
uv run marimo edit 01_neurons_and_networks.py
```

To open a different notebook, replace the filename with any `0*.py` notebook in the repo.

### With `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
marimo edit 01_neurons_and_networks.py
```

## Validation

The notebooks have been cleaned up to satisfy marimo's dependency checker.

```bash
marimo check 0*.py
python -m py_compile 0*.py
```

## Project Layout

```text
.
├── 01_neurons_and_networks.py
├── 02_training_deep_networks.py
├── 03_convolutional_networks.py
├── 04_sequence_models.py
├── 05_attention_mechanism.py
├── 06_transformer_architecture.py
├── 07_language_models.py
├── 08_modern_llm_techniques.py
└── docs/
```

## Notes

- The notebooks are designed for interactive use in `marimo`.
- Each notebook is self-contained and does not depend on local helper modules.
- Several demos use lightweight local datasets from `scikit-learn` so they can run quickly.
- The repository is educational first; the code prioritizes clarity and visual intuition over production abstractions.
