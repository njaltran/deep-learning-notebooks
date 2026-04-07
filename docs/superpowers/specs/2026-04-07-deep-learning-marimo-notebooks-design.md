# Deep Learning & LLM Interactive Marimo Notebook Series

## Context

The user is a data science student at HWR with strong classical ML/statistics foundations (regression, trees, evaluation metrics, regularization, hypothesis testing) and basic PyTorch familiarity (tensors, nn.Sequential). They have no deep learning architecture notes and need interactive learning materials covering neural network foundations through modern LLM techniques for exam/course preparation. The notebooks should bridge their existing knowledge to deep learning concepts using interactive marimo widgets.

## Overview

A series of 8 progressive marimo notebooks that teach deep learning and LLMs through interactive visualizations, mathematical explanations, and hands-on PyTorch code. Each notebook combines rich markdown exposition with interactive widgets (sliders, step-through animations, real-time visualizations) so the learner can explore concepts by manipulating parameters and observing effects.

## Project Structure

```
deep-learning-notebooks/
├── pyproject.toml                # uv project definition
├── README.md                     # Series overview and setup
├── utils/
│   └── viz.py                    # Shared visualization helpers
├── 01_neurons_and_networks.py
├── 02_training_deep_networks.py
├── 03_convolutional_networks.py
├── 04_sequence_models.py
├── 05_attention_mechanism.py
├── 06_transformer_architecture.py
├── 07_language_models.py
└── 08_modern_llm_techniques.py
```

## Dependencies

- `marimo` - notebook framework with reactive widgets
- `torch` - deep learning framework (PyTorch)
- `numpy` - numerical computation
- `matplotlib` - static plots and diagrams
- `plotly` - interactive charts (hover, zoom, pan)
- `scikit-learn` - datasets and utilities
- `transformers` - HuggingFace models (notebooks 7-8)
- `datasets` - HuggingFace datasets (notebooks 7-8)
- `peft` - Parameter-efficient fine-tuning / LoRA (notebook 8)
- `sentence-transformers` - Text embeddings for RAG (notebook 8)
- `umap-learn` - Dimensionality reduction for embedding visualization (notebook 8)

Package manager: `uv` for reproducible environment.

## Notebook Template Pattern

Every notebook follows this structure:

1. **Title + Learning Objectives** - What you'll learn (markdown)
2. **Connection to Classical ML** - Bridge from existing knowledge (e.g., "logistic regression's sigmoid is a single neuron")
3. **Concept Explanation** - Rich markdown with LaTeX math, balanced between intuition and derivations
4. **Interactive Widget** - Explore the concept by adjusting parameters
5. **Code Implementation** - Build it from scratch in PyTorch
6. **Summary + Exercises** - Key takeaways and challenges

## Notebook Details

### 01: Neurons & Networks

**Concepts:** Perceptron, activation functions (sigmoid, ReLU, tanh), multi-layer networks, forward pass, universal approximation intuition.

**Math:** Weighted sum formula, activation function definitions, matrix multiplication for layer computation.

**Bridge:** "You know logistic regression uses sigmoid to map a linear combination to [0,1]. A neuron does exactly the same thing - but we stack many of them."

**Widgets:**
- *Activation function explorer:* Sliders for weight and bias, dropdown for activation function. Plot shows the function shape updating in real-time. Side-by-side comparison of different activations.
- *Network architecture builder:* Sliders for number of layers (1-5) and neurons per layer (1-32). Renders a network diagram. Forward pass visualization with random data flowing through, showing intermediate values.
- *Decision boundary visualizer:* Train a small network on a 2D dataset (moons/circles), slider for hidden layer size, watch the decision boundary evolve.

**Code:** Build a 2-layer classifier from scratch using `nn.Module`, train on sklearn's `make_moons`.

### 02: Training Deep Networks

**Concepts:** Loss functions (MSE, cross-entropy), gradient descent, backpropagation (chain rule), optimizers (SGD, momentum, Adam), learning rate schedules, batch size effects, overfitting, dropout, batch normalization.

**Math:** Chain rule derivation for a 2-layer network, gradient update equations, Adam update rules.

**Bridge:** "MSE and R-squared from your regression notes measure prediction error. Cross-entropy does the same for classification - and it's the loss function that makes neural networks learn."

**Widgets:**
- *Gradient descent on a loss surface:* 3D plotly surface with a point showing optimizer position. Sliders for learning rate and momentum. Buttons to step or auto-run. Compare SGD vs Adam paths side-by-side.
- *Backprop step-through:* A small (2-layer, 2-neuron) network. Step forward and backward one operation at a time. Each step highlights the current computation and shows intermediate gradients.
- *Training dashboard:* Live training on MNIST. Sliders for learning rate, batch size, dropout rate. Dropdown for optimizer. Real-time loss and accuracy curves. Train/val split to show overfitting.
- *Overfitting demo:* Slider for model size (neurons) and epochs. Polynomial fitting on synthetic data. Watch train loss decrease while val loss increases.

**Code:** Full MNIST training loop with configurable hyperparameters.

### 03: Convolutional Networks

**Concepts:** 1D and 2D convolution, filters/kernels, stride, padding, pooling (max, average), feature maps, receptive field, common architectures (LeNet-style).

**Math:** Convolution formula, output size calculation: `(W - F + 2P) / S + 1`.

**Bridge:** "Feature engineering in classical ML (e.g., polynomial features) is manual. CNNs learn their own features automatically from data."

**Widgets:**
- *Convolution step-through:* A small image (e.g., 8x8 grid of numbers). A 3x3 kernel. Step through the convolution pixel by pixel, highlighting the receptive field and showing the element-wise multiplication and sum.
- *Filter gallery:* Apply preset filters (edge detection, blur, sharpen, emboss) to a sample image. See input and output side by side. Then show learned filters from a trained model.
- *Architecture builder:* Stack conv, pool, and FC layers with dropdowns. See the shape transformation at each layer (e.g., 28x28x1 -> 26x26x32 -> 13x13x32 -> ...).

**Code:** Build a CNN for Fashion-MNIST classification. Visualize learned first-layer filters.

### 04: Sequence Models (RNN/LSTM)

**Concepts:** Sequential data, vanilla RNN, hidden state, vanishing/exploding gradients, LSTM (forget gate, input gate, output gate, cell state), GRU as simplified LSTM.

**Math:** RNN hidden state equation `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)`, LSTM gate equations with sigmoid and tanh.

**Bridge:** "In classical ML, each data point is independent. RNNs model sequences where order matters - like time series or text."

**Widgets:**
- *RNN unrolling:* Animated diagram showing the network unrolled across time steps. Feed in a character sequence, watch the hidden state vector change at each step. Color-coded to show which values are large/small.
- *LSTM gate visualizer:* Interactive diagram of one LSTM cell. Sliders simulate gate values (0-1). See what fraction of the cell state is forgotten, what new info is added, and what is output. Color shows information flow.
- *Vanishing gradient demo:* Slider for sequence length (5-100). Plot gradient magnitude at each time step during backprop. Watch gradients shrink (vanilla RNN) vs remain stable (LSTM).

**Code:** Character-level language model with LSTM. Train on a small text (e.g., Shakespeare), generate text.

### 05: Attention Mechanism

**Concepts:** Limitations of fixed-size context vectors, dot-product attention, scaled dot-product attention, multi-head attention, self-attention vs cross-attention.

**Math:** Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V. Step-by-step matrix computation with actual numbers.

**Bridge:** "Think of attention as a learned, weighted average - similar to how kernel methods weight nearby points, but attention learns which 'points' (tokens) are relevant to each other."

**Widgets:**
- *Attention heatmap:* Text input field. Compute self-attention on the input tokens. Display as an interactive heatmap where hovering shows attention weights. Preset examples showing interesting patterns (e.g., pronouns attending to their referents).
- *Scaled dot-product step-through:* Small example (4 tokens, dim=3). Show Q, K, V matrices. Step through: (1) QK^T, (2) scale by sqrt(d_k), (3) softmax, (4) multiply by V. Each step shows the full matrix with color coding.
- *Multi-head comparison:* Slider for number of heads (1-8). Show how input is split across heads, each head's attention pattern, and the concatenated result. Demonstrates that different heads learn different relationships.

**Code:** Implement scaled dot-product attention and multi-head attention from scratch in PyTorch.

### 06: Transformer Architecture

**Concepts:** Encoder-decoder architecture, positional encoding (sinusoidal), layer normalization, feed-forward network, residual connections, masked self-attention, complete transformer block.

**Math:** Positional encoding formulas PE(pos,2i) = sin(pos/10000^{2i/d}), PE(pos,2i+1) = cos(pos/10000^{2i/d}). Full transformer block equations.

**Bridge:** "The transformer combines several ideas: attention (notebook 05), residual connections (like skip connections in ResNets), and layer normalization (related to batch norm from notebook 02)."

**Widgets:**
- *Positional encoding visualizer:* Heatmap of PE matrix. Sliders for sequence length and embedding dimension. Show how position information varies across dimensions. Plot individual dimensions as sine waves.
- *Transformer block data flow:* Animated step-through of one encoder block. Input tensor flows through: self-attention -> add & norm -> FFN -> add & norm. Each step shows tensor shapes and values. Highlight residual connections.
- *Encoder vs Decoder:* Side-by-side comparison showing the decoder's additional masked self-attention and cross-attention layers. Toggle causal mask on/off to see the difference.

**Code:** Build a complete transformer encoder from scratch. Train on a simple task (e.g., sequence reversal or sorting).

### 07: Language Models (GPT)

**Concepts:** Autoregressive language modeling, tokenization (BPE), token and positional embeddings, causal masking, temperature scaling, top-k and top-p (nucleus) sampling, perplexity.

**Math:** P(token|context) via softmax, temperature scaling formula, perplexity = exp(avg cross-entropy loss).

**Bridge:** "Logistic regression predicts P(y|x) for one class. A language model predicts P(next_token|all_previous_tokens) - it's classification over the entire vocabulary at each step."

**Widgets:**
- *BPE tokenizer explorer:* Text input, real-time tokenization display. Show token IDs, color-coded token boundaries, vocabulary size. Compare different tokenizers (character, word, BPE).
- *Causal mask visualizer:* NxN attention grid. Toggle mask on/off. See which tokens can attend to which. Animate token-by-token generation showing how the visible context grows.
- *Text generation playground:* Use a small pretrained model (distilgpt2). Text prompt input. Sliders for temperature (0.1-2.0), top-k (1-100), top-p (0.1-1.0). Generate button. Show probability distribution over next token, highlight sampled token. Generate multiple completions to see diversity.

**Code:** Build a mini character-level GPT. Train on a small corpus. Implement sampling strategies from scratch.

### 08: Modern LLM Techniques

**Concepts:** Transfer learning, fine-tuning, parameter-efficient fine-tuning (LoRA), prompt engineering patterns, text embeddings, vector similarity, retrieval-augmented generation (RAG), RLHF overview.

**Math:** LoRA: W' = W + BA where B is (d x r) and A is (r x d), r << d. Cosine similarity for embeddings.

**Bridge:** "Fine-tuning is like transfer learning in classical ML - you don't train from scratch, you adapt a pre-trained model. LoRA makes this efficient by only updating a small fraction of parameters."

**Widgets:**
- *LoRA parameter calculator:* Sliders for model size, layer count, and LoRA rank. Show parameter count for full fine-tuning vs LoRA. Visualize the low-rank decomposition as a matrix diagram.
- *Embedding space explorer:* Text input for multiple sentences. Compute embeddings (all-MiniLM-L6-v2). Plot in 2D/3D with UMAP/t-SNE. Drag to see nearest neighbors. Show cosine similarity scores.
- *RAG pipeline step-through:* Animated walkthrough: (1) chunk documents, (2) embed chunks, (3) user query, (4) retrieve top-k similar chunks, (5) augment prompt with context, (6) generate answer. Each step shows actual data flowing through.

**Code:** Fine-tune a small model with LoRA using HuggingFace PEFT. Build a simple RAG pipeline with sentence-transformers and a local vector store.

## Shared Utilities (`utils/viz.py`)

Reusable components to keep notebooks focused on content:

- `plot_network(layers: list[int])` - Draw a neural network architecture diagram
- `plot_attention_heatmap(weights, tokens)` - Interactive attention visualization
- `plot_loss_curves(history)` - Training/validation loss and accuracy plots
- `plot_gradient_flow(named_params)` - Gradient magnitude per layer
- `create_training_controls()` - Standard slider group for LR, batch size, epochs
- Consistent plotly theme (dark mode, colorblind-friendly palette)

## Technical Constraints

- **CPU-trainable:** All models must train in seconds on a laptop CPU. Small architectures, small datasets.
- **Self-contained:** Each notebook can be run independently (imports its own data/models).
- **Progressive references:** Later notebooks reference earlier concepts with links but don't require them to run.
- **Marimo reactivity:** Leverage marimo's cell dependency graph. Widget changes automatically trigger recomputation of dependent cells. Use `mo.ui.run_button()` for expensive operations (training loops).

## Verification

1. `uv run marimo edit <notebook>.py` - Each notebook opens and runs without errors
2. All widgets are interactive and responsive
3. Training loops complete in under 30 seconds on CPU
4. Markdown renders correctly with LaTeX math
5. Plotly charts are interactive (hover, zoom)
6. Each notebook can be run independently (no cross-notebook dependencies at runtime)
