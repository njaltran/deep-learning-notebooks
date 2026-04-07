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
    # 05 — Attention Mechanism

    ## Learning Objectives
    - Understand why attention was invented (limitations of fixed-size context)
    - Compute scaled dot-product attention step by step
    - Visualize attention weights as heatmaps
    - Explore multi-head attention and how different heads learn different patterns

    ## Connection to Classical ML
    Think of attention as a **learned, dynamic weighted average**. In KNN, you weight
    nearby points by distance. In kernel methods, you weight by similarity. Attention
    does the same thing — but the "similarity" is learned, and it operates over sequence
    positions rather than feature space. Each token asks: "which other tokens are most
    relevant to me?"

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Why Attention?

    RNNs compress an entire sequence into a fixed-size hidden state vector.
    For long sequences, this creates an **information bottleneck** — early tokens
    get "forgotten" by the time we reach the end.

    **Attention solves this by letting the model look at ALL positions simultaneously.**

    Instead of relying on a single compressed vector, attention computes a weighted
    combination of all positions, where the weights reflect relevance.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Scaled Dot-Product Attention

    The core formula:

    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

    where:
    - $Q$ (Query): "What am I looking for?"
    - $K$ (Key): "What do I contain?"
    - $V$ (Value): "What information do I provide?"

    The dot product $QK^T$ measures similarity. We scale by $\sqrt{d_k}$ to prevent
    the softmax from becoming too peaked (which would kill gradients).

    ### Step-by-Step Walkthrough

    Let's trace through the computation with actual numbers. Use the slider to
    step through each stage:
    """)
    return


@app.cell
def _(mo):
    attn_step = mo.ui.slider(
        start=0, stop=4, step=1, value=0, label="Computation Step"
    )
    attn_step
    return (attn_step,)


@app.cell
def _(attn_step, mo):
    import numpy as _np

    # Small example: 4 tokens, d_k = 3
    _np.random.seed(42)
    tokens = ["The", "cat", "sat", "down"]
    _d_k = 3

    _Q = _np.array([
        [1.0, 0.5, 0.2],
        [0.3, 1.2, 0.1],
        [0.8, 0.1, 1.0],
        [0.2, 0.9, 0.5],
    ])
    _K = _np.array([
        [0.9, 0.4, 0.3],
        [0.2, 1.1, 0.2],
        [0.7, 0.3, 0.9],
        [0.1, 0.8, 0.6],
    ])
    _V = _np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.3],
        [0.5, 0.5, 1.0],
        [0.3, 0.7, 0.2],
    ])

    # Computations
    _scores = _Q @ _K.T
    scaled_scores = _scores / _np.sqrt(_d_k)

    def softmax_rows(x):
        e = _np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    weights = softmax_rows(scaled_scores)
    output = weights @ _V

    def fmt_matrix(m, name=""):
        rows = " \\\\ ".join([" & ".join(f"{v:.2f}" for v in row) for row in m])
        return f"$${name} = \\begin{{bmatrix}} {rows} \\end{{bmatrix}}$$"

    scaled_scores_label = "\\frac{QK^T}{\\sqrt{d_k}}"
    softmax_label = "\\text{softmax}(\\cdot)"
    attention_output_label = "\\text{Attention}(Q,K,V)"

    steps_content = [
        f"""
        **Step 0: Input Matrices**

        We have 4 tokens: {tokens}. Each has a Query, Key, and Value vector of dimension {_d_k}.

        {fmt_matrix(_Q, "Q")}
        {fmt_matrix(_K, "K")}
        {fmt_matrix(_V, "V")}

        Q = "What am I looking for?" | K = "What do I contain?" | V = "What info do I give?"
        """,
        f"""
        **Step 1: Compute Raw Attention Scores ($QK^T$)**

        Dot product between each query and all keys measures similarity:

        {fmt_matrix(_scores, "QK^T")}

        Higher values = more similar. Row $i$ shows how much token $i$ attends to each other token.
        """,
        f"""
        **Step 2: Scale by $\\sqrt{{d_k}}$**

        Divide by $\\sqrt{{{_d_k}}} = {_np.sqrt(_d_k):.2f}$ to prevent extreme softmax values:

        {fmt_matrix(scaled_scores, scaled_scores_label)}

        Without scaling, large dot products → softmax close to one-hot → vanishing gradients.
        """,
        f"""
        **Step 3: Apply Softmax (row-wise)**

        Convert scores to probabilities — each row sums to 1:

        {fmt_matrix(weights, softmax_label)}

        These are the **attention weights**. Row $i$ shows the distribution of attention
        from token $i$ to all other tokens. For example, "{tokens[0]}" attends most to
        token {_np.argmax(weights[0])} ("{tokens[_np.argmax(weights[0])]}").
        """,
        f"""
        **Step 4: Weighted Sum of Values**

        Multiply attention weights by V to get the output:

        {fmt_matrix(output, attention_output_label)}

        Each output row is a **weighted combination** of all Value vectors,
        where the weights come from attention. Token representations are now
        enriched with information from relevant tokens.
        """,
    ]

    mo.md(steps_content[attn_step.value])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Attention Heatmap: Who Attends to Whom?

    Type a sentence and see the self-attention pattern. The heatmap shows how much
    each token (row) attends to every other token (column):
    """)
    return


@app.cell
def _(mo):
    attn_text_input = mo.ui.text(
        value="the cat sat on the mat",
        label="Input sentence",
    )
    attn_text_input
    return (attn_text_input,)


@app.cell
def _(attn_text_input):
    import torch as _torch
    import torch.nn as _nn
    import numpy as _np
    import plotly.graph_objects as _go

    sentence = attn_text_input.value or "the cat sat on the mat"
    words = sentence.strip().split()

    # Create simple embeddings and compute attention
    vocab = sorted(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    _d_model = 16
    _torch.manual_seed(42)
    embedding = _nn.Embedding(len(vocab), _d_model)
    W_q = _nn.Linear(_d_model, _d_model, bias=False)
    W_k = _nn.Linear(_d_model, _d_model, bias=False)

    indices = _torch.LongTensor([word_to_idx[w] for w in words])
    emb = embedding(indices)  # (seq_len, d_model)

    _Q = W_q(emb)
    _K = W_k(emb)

    _scores = (_Q @ _K.T) / _np.sqrt(_d_model)
    _attn_weights = _torch.softmax(_scores, dim=-1).detach().numpy()

    _fig = _go.Figure(
        data=_go.Heatmap(
            z=_attn_weights,
            x=words,
            y=words,
            colorscale="Viridis",
            text=_np.round(_attn_weights, 2).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.3f}<extra></extra>",
        )
    )

    _fig.update_layout(
        template="plotly_dark",
        title="Self-Attention Weights",
        xaxis_title="Key (attending to)",
        yaxis_title="Query (attending from)",
        yaxis=dict(autorange="reversed"),
        height=400,
        width=500,
        margin=dict(l=80, r=30, t=60, b=50),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Multi-Head Attention

    One attention head learns one type of relationship (e.g., syntactic proximity).
    **Multi-head attention** runs several attention heads in parallel, each learning
    different patterns:

    $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

    where each head: $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

    **Why multiple heads?**
    - Head 1 might learn "next word" relationships
    - Head 2 might learn "subject-verb" connections
    - Head 3 might learn positional patterns

    Adjust the number of heads to see different attention patterns:
    """)
    return


@app.cell
def _(mo):
    n_heads_slider = mo.ui.slider(
        start=1, stop=8, step=1, value=4, label="Number of Heads"
    )
    n_heads_slider
    return (n_heads_slider,)


@app.cell
def _(n_heads_slider):
    import torch as _torch
    import torch.nn as _nn
    import numpy as _np
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    n_heads = n_heads_slider.value
    seq_len = 6
    _d_model = 32
    _d_k = _d_model // n_heads

    mh_tokens = ["The", "quick", "brown", "fox", "jumps", "over"]

    _torch.manual_seed(42)
    # Random embeddings for demonstration
    x = _torch.randn(1, seq_len, _d_model)

    mha = _nn.MultiheadAttention(_d_model, n_heads, batch_first=True)

    with _torch.no_grad():
        _, _attn_weights = mha(x, x, x, need_weights=True, average_attn_weights=False)
        # attn_weights shape: (1, n_heads, seq_len, seq_len)
        weights_np = _attn_weights[0].numpy()

    # Show up to 4 heads
    n_show = min(n_heads, 4)
    _fig = _make_subplots(
        rows=1, cols=n_show,
        subplot_titles=[f"Head {_i+1}" for _i in range(n_show)],
        horizontal_spacing=0.05,
    )

    for _i in range(n_show):
        _fig.add_trace(
            _go.Heatmap(
                z=weights_np[_i],
                x=mh_tokens, y=mh_tokens,
                colorscale="Viridis",
                showscale=False,
                hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.3f}<extra></extra>",
            ),
            row=1, col=_i+1,
        )
        _fig.update_yaxes(autorange="reversed", row=1, col=_i+1)

    _fig.update_layout(
        template="plotly_dark",
        title=f"Multi-Head Attention: {n_heads} heads, d_model={_d_model}, d_k={_d_k}",
        height=350,
        margin=dict(l=60, r=30, t=80, b=30),
    )

    _fig
    return


@app.cell
def _(mo, n_heads_slider):
    n_h = n_heads_slider.value
    d_m = 32
    d_per_head = d_m // n_h
    params_per_head = d_m * d_per_head * 3  # Q, K, V projections
    total_params = params_per_head * n_h + d_m * d_m  # + output projection

    mo.md(
        f"""
        ### Multi-Head Attention Statistics

        | Property | Value |
        |----------|-------|
        | Total heads | {n_h} |
        | d_model | {d_m} |
        | d_k = d_v per head | {d_per_head} |
        | Parameters per head (Q+K+V) | {params_per_head:,} |
        | Output projection | {d_m * d_m:,} |
        | **Total parameters** | **{total_params:,}** |

        Notice: **total parameters stay the same** regardless of the number of heads!
        More heads = smaller per-head dimension, but the same total computation. The benefit
        is that each head can specialize in different types of relationships.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Implementing Attention from Scratch

    Here's the complete implementation in PyTorch:

    ```python
    import torch
    import torch.nn as nn
    import math

    class ScaledDotProductAttention(nn.Module):
        def forward(self, Q, K, V, mask=None):
            d_k = Q.size(-1)
            # Step 1-2: Compute scaled scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

            # Optional: apply mask (for causal/decoder attention)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Step 3: Softmax to get attention weights
            weights = torch.softmax(scores, dim=-1)

            # Step 4: Weighted sum of values
            return torch.matmul(weights, V), weights


    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

            self.attention = ScaledDotProductAttention()

        def forward(self, Q, K, V, mask=None):
            batch_size = Q.size(0)

            # Project and reshape: (batch, seq, d_model) → (batch, n_heads, seq, d_k)
            Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

            # Attention per head
            out, weights = self.attention(Q, K, V, mask)

            # Concat heads and project
            out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
            return self.W_o(out), weights
    ```

    This is the building block of the Transformer — the topic of the next notebook.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | What You Learned |
    |---------|-----------------|
    | **Attention** | Dynamic weighted average — each token selects what's relevant |
    | **Q, K, V** | Query asks, Key advertises, Value provides the actual information |
    | **Scaling** | Divide by $\sqrt{d_k}$ to keep softmax well-behaved |
    | **Multi-head** | Multiple attention patterns in parallel, same parameter budget |
    | **Self-attention** | Q, K, V all come from the same sequence |

    ### Key Takeaway
    Attention replaces sequential processing (RNN) with **parallel, direct connections**
    between all positions. Every token can attend to every other token in one step,
    regardless of distance. This is what makes Transformers so powerful — and fast.

    **Next up:** [Notebook 06 — Transformer Architecture](06_transformer_architecture.py) —
    putting attention together with feed-forward layers, normalization, and positional encoding.
    """)
    return


if __name__ == "__main__":
    app.run()
