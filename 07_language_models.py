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
    # 07 — Language Models (GPT)

    ## Learning Objectives
    - Understand autoregressive language modeling ("predict the next token")
    - Explore tokenization and why BPE matters
    - See how causal masking enables generation
    - Play with temperature, top-k, and top-p sampling
    - Build a mini character-level GPT

    ## Connection to Classical ML
    Logistic regression predicts $P(y | x)$ for a single class.
    A language model predicts $P(\text{next\_token} | \text{all\_previous\_tokens})$ —
    it's **classification over the entire vocabulary** at each step. The "features" are the
    preceding context, and the "classes" are all possible next tokens.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Autoregressive Language Modeling

    A language model learns the probability distribution:

    $$P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t | x_1, \ldots, x_{t-1})$$

    At each step, the model sees all previous tokens and predicts the next one.
    This is the chain rule of probability — decomposing a joint distribution into
    conditional distributions.

    **Training:** Given text "The cat sat on the mat", predict each next token:
    - Input: "The" → Predict: "cat"
    - Input: "The cat" → Predict: "sat"
    - Input: "The cat sat" → Predict: "on"
    - ...

    **Generation:** Start with a prompt, sample the next token, append it, repeat.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Tokenization: From Text to Numbers

    Neural networks work with numbers, not text. **Tokenization** converts text into
    a sequence of integer IDs. There are several strategies:

    | Method | Example: "unhappiness" | Vocab Size | Trade-off |
    |--------|----------------------|------------|-----------|
    | **Character** | u, n, h, a, p, p, i, n, e, s, s | ~100 | Long sequences, simple |
    | **Word** | unhappiness | ~100K+ | Short sequences, huge vocab |
    | **BPE (subword)** | un, happi, ness | ~30K-50K | Best of both worlds |

    **BPE (Byte Pair Encoding)** is what GPT uses. It learns common character pairs
    and merges them iteratively, creating a vocabulary of subword units.

    ### Tokenizer Explorer

    Type text to see how it gets tokenized:
    """)
    return


@app.cell
def _(mo):
    tokenizer_input = mo.ui.text_area(
        value="The transformer architecture revolutionized natural language processing.",
        label="Enter text to tokenize",
        max_length=500,
    )
    tokenizer_method = mo.ui.dropdown(
        options=["Character", "Word (whitespace)", "Simple BPE (simulated)"],
        value="Simple BPE (simulated)",
        label="Tokenization Method",
    )
    mo.hstack([tokenizer_input, tokenizer_method], justify="center", gap=1)
    return tokenizer_input, tokenizer_method


@app.cell
def _(mo, tokenizer_input, tokenizer_method):
    text = tokenizer_input.value or "Hello world"
    method = tokenizer_method.value

    if method == "Character":
        _tokens = list(text)
        vocab = sorted(set(_tokens))
    elif method == "Word (whitespace)":
        _tokens = text.split()
        vocab = sorted(set(_tokens))
    else:  # Simple BPE simulation
        # Simulate BPE by breaking into common subword units
        common_units = [
            "the", "ing", "tion", "er", "ed", "al", "ize", "re", "un",
            "pre", "dis", "ment", "ness", "ous", "ful", "able", "ible",
            "ly", "The", "an", "ar", "at", "en", "es", "in", "is", "on",
            "or", "st", "ur", "ll", "ss", "tt",
        ]
        _tokens = []
        remaining = text
        while remaining:
            matched = False
            # Try longest match first
            for length in range(min(6, len(remaining)), 0, -1):
                chunk = remaining[:length]
                if chunk in common_units or chunk == " " or (length == 1 and not chunk.isalpha()):
                    _tokens.append(chunk)
                    remaining = remaining[length:]
                    matched = True
                    break
            if not matched:
                _tokens.append(remaining[0])
                remaining = remaining[1:]
        vocab = sorted(set(_tokens))

    # Color-code tokens
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]

    colored_tokens = ""
    for _i, token in enumerate(_tokens):
        color = colors[_i % len(colors)]
        display = token.replace(" ", "·")
        colored_tokens += f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; border-radius: 3px; color: white; font-family: monospace;">{display}</span> '

    token_ids = {t: _i for _i, t in enumerate(vocab)}
    ids_display = [str(token_ids[t]) for t in _tokens]

    mo.md(
        f"""
        ### Tokenization Result

        **Method:** {method}

        **Tokens ({len(_tokens)}):**
        {colored_tokens}

        **Token IDs:** [{', '.join(ids_display)}]

        **Vocabulary size:** {len(vocab)} unique tokens

        ---
        **Key insight:** BPE balances sequence length and vocabulary size.
        Character-level creates {len(list(text))} tokens for this text.
        Word-level creates {len(text.split())} tokens but needs a massive vocabulary.
        BPE creates {len(_tokens)} tokens with a manageable vocabulary.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Causal Masking for Generation

    During training, the model processes the entire sequence at once (for parallelism).
    But each position should only attend to **previous** positions — otherwise it could
    "cheat" by looking at the answer.

    The **causal mask** enforces this:
    """)
    return


@app.cell
def _(mo):
    mask_gen_step = mo.ui.slider(
        start=1, stop=8, step=1, value=1, label="Generation Step"
    )
    mask_gen_step
    return (mask_gen_step,)


@app.cell
def _(mask_gen_step, mo):
    import numpy as _np
    import plotly.graph_objects as _go

    _generation_step = mask_gen_step.value
    _tokens = ["<start>", "The", "cat", "sat", "on", "the", "mat", "<end>"]
    n = len(_tokens)

    # Build mask showing visible context at current step
    mask = _np.zeros((n, n))
    for _i in range(min(_generation_step + 1, n)):
        for _j in range(_i + 1):
            mask[_i, _j] = 1

    # Highlight current prediction
    annotations = []
    if _generation_step < n:
        annotations.append(dict(
            x=_generation_step, y=_generation_step,
            text="PREDICT",
            showarrow=False,
            font=dict(color="white", size=10),
        ))

    _fig = _go.Figure(
        data=_go.Heatmap(
            z=mask[::-1],
            x=_tokens, y=_tokens[::-1],
            colorscale=[[0, "#1a1a2e"], [0.5, "#2a2a4e"], [1, "#636EFA"]],
            showscale=False,
        )
    )

    _fig.update_layout(
        template="plotly_dark",
        title=f"Generation Step {_generation_step}: Predicting '{_tokens[min(_generation_step, n-1)]}'",
        xaxis_title="Tokens in context (can see these)",
        yaxis_title="Position predicting from",
        height=400, width=450,
        annotations=annotations,
        margin=dict(l=80, r=30, t=60, b=50),
    )

    visible = _tokens[:_generation_step]
    next_token = _tokens[min(_generation_step, n-1)]

    mo.md(
        f"""
        **Step {_generation_step}:** The model sees: **{' '.join(visible) if visible else '(nothing)'}**

        It must predict: **{next_token}**

        The bright cells show which positions can attend to which.
        Each position can only see itself and everything before it.
        """
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Sampling Strategies: Temperature, Top-k, Top-p

    After the model outputs a probability distribution over the vocabulary, we need
    to **sample** the next token. Different strategies produce different text quality:

    ### Temperature Scaling

    $$P(token_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

    - $T = 1$: Standard softmax (original probabilities)
    - $T < 1$: Sharper distribution → more deterministic, repetitive
    - $T > 1$: Flatter distribution → more random, creative

    ### Top-k Sampling
    Only consider the top $k$ most likely tokens. Eliminates unlikely tokens.

    ### Top-p (Nucleus) Sampling
    Include tokens until cumulative probability reaches $p$. Adapts the number
    of candidates to the model's confidence.
    """)
    return


@app.cell
def _(mo):
    temp_slider = mo.ui.slider(
        start=0.1, stop=3.0, step=0.1, value=1.0, label="Temperature"
    )
    topk_slider = mo.ui.slider(
        start=1, stop=20, step=1, value=10, label="Top-k"
    )
    topp_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.9, label="Top-p"
    )
    mo.hstack([temp_slider, topk_slider, topp_slider], justify="center", gap=1)
    return temp_slider, topk_slider, topp_slider


@app.cell
def _(temp_slider, topk_slider, topp_slider):
    import numpy as _np
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    # Simulated logits for vocabulary
    _np.random.seed(42)
    vocab_words = ["the", "a", "cat", "dog", "sat", "on", "mat", "ran",
                   "quickly", "slowly", "big", "small", "happy", "red",
                   "blue", "green", "table", "chair", "house", "tree"]
    raw_logits = _np.array([3.5, 2.1, 1.8, 1.5, 1.2, 0.9, 0.7, 0.5,
                          0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4,
                          -0.5, -0.7, -0.9, -1.2])

    # Apply temperature
    T = temp_slider.value
    scaled_logits = raw_logits / T
    _base_probs = _np.exp(scaled_logits - scaled_logits.max())
    _base_probs = _base_probs / _base_probs.sum()

    # Apply top-k
    k = topk_slider.value
    topk_mask = _np.zeros_like(_base_probs)
    topk_indices = _np.argsort(_base_probs)[-k:]
    topk_mask[topk_indices] = _base_probs[topk_indices]
    topk_probs = topk_mask / topk_mask.sum()

    # Apply top-p
    p = topp_slider.value
    sorted_indices = _np.argsort(_base_probs)[::-1]
    sorted_probs = _base_probs[sorted_indices]
    cumsum = _np.cumsum(sorted_probs)
    cutoff = _np.searchsorted(cumsum, p) + 1
    topp_mask = _np.zeros_like(_base_probs)
    topp_mask[sorted_indices[:cutoff]] = _base_probs[sorted_indices[:cutoff]]
    topp_probs = topp_mask / topp_mask.sum()

    _fig = _make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"After Temperature ({T})",
            f"After Top-k (k={k})",
            f"After Top-p (p={p})",
        ],
    )

    _fig.add_trace(
        _go.Bar(x=vocab_words, y=_base_probs, marker_color="#636EFA", showlegend=False),
        row=1, col=1,
    )
    _fig.add_trace(
        _go.Bar(
            x=vocab_words, y=topk_probs,
            marker_color=["#636EFA" if topk_probs[i] > 0 else "#333" for i in range(len(vocab_words))],
            showlegend=False,
        ),
        row=1, col=2,
    )
    _fig.add_trace(
        _go.Bar(
            x=vocab_words, y=topp_probs,
            marker_color=["#00CC96" if topp_probs[i] > 0 else "#333" for i in range(len(vocab_words))],
            showlegend=False,
        ),
        row=1, col=3,
    )

    _fig.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=50, r=30, t=60, b=80),
    )
    _fig.update_xaxes(tickangle=45)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Build a Mini Character-Level GPT

    Let's build a small GPT-style model from scratch and train it to generate text:
    """)
    return


@app.cell
def _(mo):
    gpt_train_btn = mo.ui.run_button(label="Train Mini GPT")
    gpt_temp = mo.ui.slider(start=0.3, stop=2.0, step=0.1, value=0.8, label="Generation Temperature")
    gpt_prompt = mo.ui.text(value="To be ", label="Prompt", max_length=30)
    mo.hstack([gpt_prompt, gpt_temp, gpt_train_btn], justify="center", gap=1)
    return gpt_prompt, gpt_temp, gpt_train_btn


@app.cell
def _(gpt_prompt, gpt_temp, gpt_train_btn, mo):
    import torch as _torch
    import torch.nn as _nn
    import plotly.graph_objects as _go

    gpt_train_btn.value

    # Corpus
    corpus = """To be or not to be that is the question
    Whether tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune
    Or to take arms against a sea of troubles
    And by opposing end them To die to sleep
    No more and by a sleep to say we end
    The heartache and the thousand natural shocks
    That flesh is heir to Tis a consummation
    Devoutly to be wished To die to sleep
    To sleep perchance to dream ay theres the rub
    For in that sleep of death what dreams may come
    When we have shuffled off this mortal coil
    Must give us pause theres the respect
    That makes calamity of so long life"""

    chars = sorted(set(corpus))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    # Mini GPT
    class MiniGPT(_nn.Module):
        def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
            super().__init__()
            self.block_size = block_size
            self.tok_embed = _nn.Embedding(vocab_size, d_model)
            self.pos_embed = _nn.Embedding(block_size, d_model)
            self.blocks = _nn.ModuleList([
                _nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                    dropout=0.1, batch_first=True, norm_first=True,
                )
                for _ in range(n_layers)
            ])
            self.ln_f = _nn.LayerNorm(d_model)
            self.head = _nn.Linear(d_model, vocab_size)

        def forward(self, x):
            B, T = x.shape
            tok = self.tok_embed(x)
            pos = self.pos_embed(_torch.arange(T, device=x.device))
            x = tok + pos
            # Causal mask
            mask = _torch.triu(_torch.ones(T, T), diagonal=1).bool()
            for block in self.blocks:
                x = block(x, src_mask=mask, is_causal=True)
            x = self.ln_f(x)
            return self.head(x)

    block_size = 32
    d_model = 64
    n_heads = 4
    n_layers = 2

    model = MiniGPT(vocab_size, d_model, n_heads, n_layers, block_size)
    optimizer = _torch.optim.Adam(model.parameters(), lr=0.003)

    # Prepare training data
    data = _torch.LongTensor([char_to_idx[c] for c in corpus])

    losses = []
    for _train_step in range(300):
        # Random batch of subsequences
        starts = _torch.randint(0, len(data) - block_size - 1, (16,))
        xb = _torch.stack([data[s:s+block_size] for s in starts])
        yb = _torch.stack([data[s+1:s+block_size+1] for s in starts])

        logits = model(xb)
        loss = _nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Generate
    model.eval()
    prompt = gpt_prompt.value or "To "
    # Encode prompt (only use chars that exist in vocab)
    prompt_chars = [c for c in prompt if c in char_to_idx]
    context = _torch.LongTensor([[char_to_idx[c] for c in prompt_chars]])
    generated = "".join(prompt_chars)

    with _torch.no_grad():
        for _ in range(200):
            ctx = context[:, -block_size:]
            logits = model(ctx)
            next_logits = logits[0, -1, :] / gpt_temp.value
            _probs = _torch.softmax(next_logits, dim=-1)
            next_idx = _torch.multinomial(_probs, 1)
            context = _torch.cat([context, next_idx.unsqueeze(0)], dim=1)
            generated += idx_to_char[next_idx.item()]

    _fig = _go.Figure()
    _fig.add_trace(_go.Scatter(
        x=list(range(len(losses))), y=losses,
        name="Training Loss", line=dict(color="#636EFA"),
    ))
    _fig.update_layout(
        template="plotly_dark",
        title=f"Mini GPT Training | {sum(p.numel() for p in model.parameters()):,} parameters",
        xaxis_title="Step", yaxis_title="Cross-Entropy Loss",
        height=250,
        margin=dict(l=50, r=30, t=60, b=50),
    )

    mo.md(
        f"""
        ### Generated Text (temp={gpt_temp.value})

        ```
        {generated}
        ```

        **Model:** {n_layers} layers, {n_heads} heads, d_model={d_model}
        | Vocab: {vocab_size} chars | Block size: {block_size} | Final loss: {losses[-1]:.3f}

        This tiny model learns character patterns from Shakespeare. With more data and
        parameters, the same architecture scales to GPT-3/4 with billions of parameters.
        """
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. From Mini GPT to Real LLMs

    The architecture you just built is **the same** as GPT-2, GPT-3, GPT-4 — just smaller.
    The scaling recipe:

    | Model | Parameters | Layers | d_model | Heads | Training Data |
    |-------|-----------|--------|---------|-------|---------------|
    | Our Mini GPT | ~50K | 2 | 64 | 4 | ~700 chars |
    | GPT-2 Small | 117M | 12 | 768 | 12 | 40GB text |
    | GPT-3 | 175B | 96 | 12288 | 96 | 570GB text |
    | GPT-4 | ~1.8T* | ~120* | ~? | ~? | ~13T tokens* |

    *Estimated, not officially disclosed.

    The fundamental insight: **the same architecture, scaled up with more data, produces
    dramatically more capable models.** This is the "scaling law" that drives modern AI.

    ---

    ## Summary

    | Concept | What You Learned |
    |---------|-----------------|
    | **Autoregressive LM** | Predict next token given all previous tokens |
    | **Tokenization** | BPE balances vocab size and sequence length |
    | **Causal masking** | Prevents looking ahead during training |
    | **Temperature** | Controls randomness: low = deterministic, high = creative |
    | **Top-k / Top-p** | Truncate the distribution to avoid unlikely tokens |
    | **GPT architecture** | Decoder-only transformer with causal masking |

    **Next up:** [Notebook 08 — Modern LLM Techniques](08_modern_llm_techniques.py) —
    fine-tuning, LoRA, embeddings, and RAG.
    """)
    return


if __name__ == "__main__":
    app.run()
