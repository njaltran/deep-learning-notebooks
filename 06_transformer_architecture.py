import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # 06 — Transformer Architecture

    ## Learning Objectives
    - Understand positional encoding and why transformers need it
    - Trace data flow through a complete transformer block
    - Compare encoder vs decoder (masked vs unmasked attention)
    - Build a working transformer from scratch
    - Train it on a simple sequence task

    ## Connection to Previous Notebooks
    The transformer combines several ideas you already know:
    - **Attention** (Notebook 05) — the core mechanism
    - **Residual connections** — like skip connections; add the input back to the output
    - **Layer normalization** — related to batch norm (Notebook 02)
    - **Feed-forward networks** — just an MLP applied to each position independently

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Positional Encoding

    Attention treats its input as a **set** — it has no notion of order. But word order
    matters ("dog bites man" ≠ "man bites dog"). Positional encoding injects position
    information into the input embeddings.

    The original transformer uses sinusoidal encodings:

    $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
    $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

    Each position gets a unique pattern. Different dimensions capture different frequencies,
    allowing the model to learn both absolute and relative positions.
    """)
    return


@app.cell
def _(mo):
    pe_seq_len = mo.ui.slider(
        start=10, stop=100, step=10, value=50, label="Sequence Length"
    )
    pe_d_model = mo.ui.slider(
        start=16, stop=128, step=16, value=64, label="Embedding Dimension (d_model)"
    )
    mo.hstack([pe_seq_len, pe_d_model], justify="center", gap=1.5)
    return pe_d_model, pe_seq_len


@app.cell
def _(pe_d_model, pe_seq_len):
    import numpy as _np
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    _seq_len = pe_seq_len.value
    _d_model = pe_d_model.value

    # Compute positional encoding
    pe = _np.zeros((_seq_len, _d_model))
    position = _np.arange(_seq_len)[:, _np.newaxis]
    div_term = _np.exp(_np.arange(0, _d_model, 2) * -(_np.log(10000.0) / _d_model))

    pe[:, 0::2] = _np.sin(position * div_term)
    pe[:, 1::2] = _np.cos(position * div_term)

    _fig = _make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Positional Encoding Heatmap",
            "Individual Dimensions (sin waves)",
        ],
        column_widths=[0.6, 0.4],
    )

    _fig.add_trace(
        _go.Heatmap(
            z=pe, colorscale="RdBu", zmid=0,
            hovertemplate="Position: %{y}<br>Dimension: %{x}<br>Value: %{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Show a few individual dimensions
    for dim in [0, 2, 4, 8]:
        if dim < _d_model:
            _fig.add_trace(
                _go.Scatter(
                    x=list(range(_seq_len)), y=pe[:, dim],
                    name=f"dim {dim}",
                    mode="lines",
                ),
                row=1, col=2,
            )

    _fig.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    _fig.update_xaxes(title_text="Dimension", row=1, col=1)
    _fig.update_yaxes(title_text="Position", row=1, col=1)
    _fig.update_xaxes(title_text="Position", row=1, col=2)
    _fig.update_yaxes(title_text="PE value", row=1, col=2)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Notice:** Lower dimensions have high-frequency waves (change quickly with position),
    while higher dimensions have low-frequency waves (change slowly). This creates a
    unique "fingerprint" for each position — like a binary number with continuous values.

    ---

    ## 2. Transformer Block: Data Flow Step-Through

    A single transformer encoder block:

    ```
    Input → [Multi-Head Attention] → Add & Norm → [Feed-Forward] → Add & Norm → Output
       └──────────────────────────────┘        └─────────────────────────────┘
              Residual Connection                    Residual Connection
    ```

    Step through each operation:
    """)
    return


@app.cell
def _(mo):
    tf_step = mo.ui.slider(
        start=0, stop=5, step=1, value=0, label="Transformer Block Step"
    )
    tf_step
    return (tf_step,)


@app.cell
def _(mo, tf_step):
    steps = [
        {
            "title": "Step 0: Input",
            "desc": r"""
            Input: Token embeddings + positional encodings.
            Shape: `(batch, seq_len, d_model)`

            $$X = \text{TokenEmbed}(\text{tokens}) + \text{PE}$$

            The input is the same representation entering every block. In deeper transformers,
            this is the output of the previous block.
            """,
            "shape": "(B, S, 512)",
            "highlight": "input",
        },
        {
            "title": "Step 1: Multi-Head Self-Attention",
            "desc": r"""
            Compute attention over all positions simultaneously:

            $$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

            Each head projects Q, K, V separately, computes attention, then results are concatenated.
            This is where the model learns which tokens are relevant to each other.
            """,
            "shape": "(B, S, 512) → (B, S, 512)",
            "highlight": "attention",
        },
        {
            "title": "Step 2: Add & Layer Norm (Post-Attention)",
            "desc": r"""
            **Residual connection** + **Layer Normalization**:

            $$X' = \text{LayerNorm}(X + \text{MHA}(X))$$

            The residual connection ($+ X$) ensures gradients flow easily through deep stacks.
            Layer norm normalizes across the feature dimension (not the batch dimension like
            batch norm).

            Layer norm per position: $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
            """,
            "shape": "(B, S, 512) → (B, S, 512)",
            "highlight": "addnorm1",
        },
        {
            "title": "Step 3: Feed-Forward Network",
            "desc": r"""
            A simple 2-layer MLP applied to each position **independently**:

            $$\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$$

            Typical expansion: $d_{model} \rightarrow 4 \times d_{model} \rightarrow d_{model}$
            (e.g., 512 → 2048 → 512).

            This is where the model does "computation" — attention gathers information,
            FFN processes it.
            """,
            "shape": "(B, S, 512) → (B, S, 2048) → (B, S, 512)",
            "highlight": "ffn",
        },
        {
            "title": "Step 4: Add & Layer Norm (Post-FFN)",
            "desc": r"""
            Another residual connection + layer norm:

            $$\text{Output} = \text{LayerNorm}(X' + \text{FFN}(X'))$$

            The output has the same shape as the input and can be fed into the next
            transformer block.
            """,
            "shape": "(B, S, 512) → (B, S, 512)",
            "highlight": "addnorm2",
        },
        {
            "title": "Step 5: Complete Block",
            "desc": r"""
            **Putting it all together:**

            $$X' = \text{LayerNorm}(X + \text{MHA}(X))$$
            $$\text{Output} = \text{LayerNorm}(X' + \text{FFN}(X'))$$

            That's one transformer block! Stack $N$ of these (typically 6-12) for
            a full transformer encoder.

            **Parameter count for one block** (d_model=512, 8 heads, FFN=2048):
            - MHA: $4 \times 512^2 = 1,048,576$ (Q, K, V, O projections)
            - FFN: $512 \times 2048 \times 2 = 2,097,152$
            - Layer Norms: $2 \times 2 \times 512 = 2,048$
            - **Total: ~3.1M per block**
            """,
            "highlight": "all",
        },
    ]

    current_step = steps[tf_step.value]

    # Visual block diagram using text
    components = {
        "input": "→ **[INPUT]**",
        "attention": "→ **[MULTI-HEAD ATTENTION]**",
        "addnorm1": "→ **[ADD & NORM]**",
        "ffn": "→ **[FEED-FORWARD]**",
        "addnorm2": "→ **[ADD & NORM]**",
        "all": "",
    }

    block_diagram = ""
    for key, label in [("input", "Input"), ("attention", "Multi-Head Attention"),
                        ("addnorm1", "Add & Norm"), ("ffn", "Feed-Forward"),
                        ("addnorm2", "Add & Norm")]:
        marker = " ◀ **YOU ARE HERE**" if key == current_step["highlight"] else ""
        block_diagram += f"| {label} |{marker}\n"

    mo.md(
        f"""
        ### {current_step['title']}

        ```
        Shape: {current_step.get('shape', '')}
        ```

        {current_step['desc']}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Encoder vs Decoder

    The original transformer has both an **encoder** (processes input) and **decoder**
    (generates output). The key difference:

    | Feature | Encoder | Decoder |
    |---------|---------|---------|
    | Self-attention | Full (bidirectional) | **Masked** (causal — can only look backward) |
    | Cross-attention | None | Yes (attends to encoder output) |
    | Use case | Understanding (BERT) | Generation (GPT) |

    ### Causal Mask Visualization

    The mask prevents the decoder from "cheating" by looking at future tokens during generation:
    """)
    return


@app.cell
def _(mo):
    mask_seq_len = mo.ui.slider(
        start=4, stop=12, step=1, value=6, label="Sequence Length"
    )
    mask_type = mo.ui.dropdown(
        options=["No Mask (Encoder)", "Causal Mask (Decoder)"],
        value="Causal Mask (Decoder)",
        label="Mask Type",
    )
    mo.hstack([mask_seq_len, mask_type], justify="center", gap=1.5)
    return mask_seq_len, mask_type


@app.cell
def _(mask_seq_len, mask_type):
    import numpy as _np
    import plotly.graph_objects as _go

    n = mask_seq_len.value
    labels = [f"t={i}" for i in range(n)]

    if mask_type.value == "Causal Mask (Decoder)":
        # Lower triangular: can only attend to current and past
        mask = _np.tril(_np.ones((n, n)))
    else:
        mask = _np.ones((n, n))

    _fig = _go.Figure(
        data=_go.Heatmap(
            z=mask[::-1],
            x=labels, y=labels[::-1],
            colorscale=[[0, "#1a1a2e"], [1, "#636EFA"]],
            showscale=False,
            text=_np.where(mask[::-1] == 1, "✓ can attend", "✗ masked")[::-1],
            texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )

    _fig.update_layout(
        template="plotly_dark",
        title=f"{mask_type.value} | Seq length = {n}",
        xaxis_title="Key position (attending TO)",
        yaxis_title="Query position (attending FROM)",
        height=400, width=450,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **With the causal mask:** Token at position $t$ can only attend to positions $\leq t$.
    This is essential for autoregressive generation — when predicting the next word,
    you shouldn't be able to see future words.

    **GPT = Decoder only** (causal mask). **BERT = Encoder only** (no mask, bidirectional).

    ---

    ## 4. Building a Transformer from Scratch

    Let's build a minimal transformer encoder and train it on a **sequence sorting task**:
    given a shuffled sequence of numbers, output them in sorted order.
    """)
    return


@app.cell
def _(mo):
    tf_train_btn = mo.ui.run_button(label="Train Transformer")
    tf_n_layers = mo.ui.slider(start=1, stop=4, step=1, value=2, label="Layers")
    tf_n_heads = mo.ui.slider(start=1, stop=4, step=1, value=2, label="Heads")
    mo.hstack([tf_n_layers, tf_n_heads, tf_train_btn], justify="center", gap=1)
    return tf_n_heads, tf_n_layers, tf_train_btn


@app.cell
def _(mo, tf_n_heads, tf_n_layers, tf_train_btn):
    import torch as _torch
    import torch.nn as _nn
    import math
    import plotly.graph_objects as _go

    tf_train_btn.value

    # Transformer components from scratch
    class PositionalEncoding(_nn.Module):
        def __init__(self, d_model, max_len=100):
            super().__init__()
            pe = _torch.zeros(max_len, d_model)
            position = _torch.arange(max_len).unsqueeze(1).float()
            div_term = _torch.exp(_torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = _torch.sin(position * div_term)
            pe[:, 1::2] = _torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerBlock(_nn.Module):
        def __init__(self, d_model, n_heads, d_ff):
            super().__init__()
            self.attn = _nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.ffn = _nn.Sequential(
                _nn.Linear(d_model, d_ff),
                _nn.ReLU(),
                _nn.Linear(d_ff, d_model),
            )
            self.norm1 = _nn.LayerNorm(d_model)
            self.norm2 = _nn.LayerNorm(d_model)

        def forward(self, x):
            # Self-attention + residual + norm
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            # FFN + residual + norm
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x

    class SortingTransformer(_nn.Module):
        def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff):
            super().__init__()
            self.embed = _nn.Embedding(vocab_size, d_model)
            self.pe = PositionalEncoding(d_model)
            self.blocks = _nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
            ])
            self.output = _nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.pe(self.embed(x))
            for block in self.blocks:
                x = block(x)
            return self.output(x)

    # Task: sort sequences of digits 0-9
    vocab_size = 10
    _seq_len = 6
    _d_model = 32
    n_heads = tf_n_heads.value
    n_layers = tf_n_layers.value
    d_ff = 64

    model = SortingTransformer(vocab_size, _d_model, n_heads, n_layers, d_ff)
    optimizer = _torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = _nn.CrossEntropyLoss()

    # Generate training data
    def make_batch(batch_size):
        # Random sequences of digits
        x = _torch.randint(0, vocab_size, (batch_size, _seq_len))
        # Target: sorted version
        y, _ = _torch.sort(x, dim=1)
        return x, y

    # Train
    losses = []
    accuracies = []
    for _train_step in range(500):
        x_batch, y_batch = make_batch(32)
        logits = model(x_batch)  # (32, 6, 10)
        loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if _train_step % 10 == 0:
            losses.append(loss.item())
            with _torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == y_batch).float().mean().item()
                accuracies.append(acc)

    # Test
    model.eval()
    x_test, y_test = make_batch(5)
    with _torch.no_grad():
        pred = model(x_test).argmax(dim=-1)

    results_table = ""
    for i in range(5):
        inp = x_test[i].tolist()
        target = y_test[i].tolist()
        predicted = pred[i].tolist()
        correct = "✓" if predicted == target else "✗"
        results_table += f"| {inp} | {target} | {predicted} | {correct} |\n"

    _fig = _go.Figure()
    _fig.add_trace(_go.Scatter(
        x=list(range(0, 500, 10)), y=accuracies,
        name="Token Accuracy", line=dict(color="#636EFA"),
    ))
    _fig.update_layout(
        template="plotly_dark",
        title=f"Sorting Transformer | {n_layers} layers, {n_heads} heads | Final acc: {accuracies[-1]:.1%}",
        xaxis_title="Step", yaxis_title="Accuracy",
        height=300,
        margin=dict(l=50, r=30, t=60, b=50),
    )

    n_params = sum(p.numel() for p in model.parameters())

    mo.md(
        f"""
        ### Test Results ({n_params:,} parameters)

        | Input | Target (sorted) | Predicted | Correct |
        |-------|----------------|-----------|---------|
        {results_table}

        Try adjusting layers and heads — more layers generally helps, but for this simple
        task even 1-2 layers should work well.
        """
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | What You Learned |
    |---------|-----------------|
    | **Positional encoding** | Sinusoidal patterns that inject position information |
    | **Transformer block** | Self-Attention → Add & Norm → FFN → Add & Norm |
    | **Residual connections** | Skip connections that help gradient flow in deep stacks |
    | **Layer normalization** | Normalize across features (not batch) for stable training |
    | **Causal mask** | Prevents looking ahead — essential for autoregressive models |
    | **Encoder vs Decoder** | Encoder: bidirectional (BERT). Decoder: causal (GPT) |

    ### Key Takeaway
    The transformer is an elegant composition of simple parts: attention for gathering
    information, FFN for processing it, residual connections for gradient flow, and
    layer norm for stability. Stack these blocks and you get the architecture behind
    GPT, BERT, and every modern LLM.

    **Next up:** [Notebook 07 — Language Models](07_language_models.py) —
    how to use the transformer for language modeling and text generation.
    """)
    return


if __name__ == "__main__":
    app.run()
