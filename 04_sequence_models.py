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
    # 04 — Sequence Models (RNN / LSTM)

    ## Learning Objectives
    - Understand why sequential data needs special architectures
    - Visualize RNN unrolling and hidden state evolution
    - See the vanishing gradient problem and why LSTMs solve it
    - Interactively explore LSTM gates (forget, input, output)
    - Build a character-level text generator

    ## Connection to Classical ML
    In classical ML, each data point is **independent** — row 5 in your dataset has no
    relationship to row 4. But many real-world problems are **sequential**: words in a sentence,
    stock prices over time, patient vitals during a hospital stay. RNNs process sequences
    by maintaining a **hidden state** — a memory that evolves as each element is processed.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The Vanilla RNN

    At each time step $t$, the RNN updates its hidden state:

    $$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$
    $$y_t = W_{hy} \cdot h_t + b_y$$

    The same weights $W_{hh}$, $W_{xh}$ are **shared across all time steps** — this is
    what makes RNNs parameter-efficient for sequences of any length.

    Think of $h_t$ as a **compressed summary** of everything the network has seen so far.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. RNN Unrolling: Hidden State Evolution

    Feed a sequence of characters into an RNN and watch the hidden state change.
    Each color in the hidden state vector represents one dimension — watch how
    the pattern shifts as each new character is processed:
    """)
    return


@app.cell
def _(mo):
    rnn_input_text = mo.ui.text(
        value="hello world",
        label="Input sequence",
        max_length=20,
    )
    rnn_hidden_dim = mo.ui.slider(
        start=4, stop=16, step=2, value=8, label="Hidden dimension"
    )
    mo.hstack([rnn_input_text, rnn_hidden_dim], justify="center", gap=1.5)
    return rnn_hidden_dim, rnn_input_text


@app.cell
def _(rnn_hidden_dim, rnn_input_text):
    import torch as _torch
    import torch.nn as _nn
    import numpy as _np
    import plotly.graph_objects as _go

    text = rnn_input_text.value or "hello"
    hidden_dim = rnn_hidden_dim.value

    # Character-level encoding
    _chars = sorted(set(text))
    _char_to_idx = {c: i for i, c in enumerate(_chars)}
    _vocab_size = len(_chars)

    # Create RNN
    rnn = _nn.RNN(input_size=_vocab_size, hidden_size=hidden_dim, batch_first=True)

    # One-hot encode
    indices = [_char_to_idx[c] for c in text]
    _x = _torch.zeros(1, len(text), _vocab_size)
    for _t, _idx in enumerate(indices):
        _x[0, _t, _idx] = 1.0

    # Run and capture hidden states
    with _torch.no_grad():
        hidden_states = []
        h = _torch.zeros(1, 1, hidden_dim)
        for _t in range(len(text)):
            _, h = rnn(_x[:, _t:_t+1, :], h)
            hidden_states.append(h.squeeze().numpy().copy())

    hidden_matrix = _np.array(hidden_states)  # (seq_len, hidden_dim)

    _fig = _go.Figure(
        data=_go.Heatmap(
            z=hidden_matrix.T,
            x=list(text),
            y=[f"h[{_i}]" for _i in range(hidden_dim)],
            colorscale="RdBu",
            zmid=0,
            hovertemplate="Char: %{x}<br>Dim: %{y}<br>Value: %{z:.3f}<extra></extra>",
        )
    )

    _fig.update_layout(
        template="plotly_dark",
        title=f"Hidden State Evolution | '{text}' | dim={hidden_dim}",
        xaxis_title="Sequence Position (character)",
        yaxis_title="Hidden State Dimension",
        height=350,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The Vanishing Gradient Problem

    When we backpropagate through many time steps, gradients get multiplied by $W_{hh}$
    at each step. If $\|W_{hh}\| < 1$, gradients **shrink exponentially** — the network
    can't learn long-range dependencies.

    $$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_T} \cdot \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

    Each factor $\frac{\partial h_t}{\partial h_{t-1}}$ involves $W_{hh}$ and the activation
    derivative. Multiply many small numbers → vanishing gradient.

    Adjust the sequence length below and see how gradient magnitude decays:
    """)
    return


@app.cell
def _(mo):
    vanish_seq_len = mo.ui.slider(
        start=5, stop=50, step=5, value=20, label="Sequence Length"
    )
    vanish_model_type = mo.ui.dropdown(
        options=["Vanilla RNN", "LSTM"],
        value="Vanilla RNN",
        label="Model Type",
    )
    mo.hstack([vanish_seq_len, vanish_model_type], justify="center", gap=1.5)
    return vanish_model_type, vanish_seq_len


@app.cell
def _(vanish_model_type, vanish_seq_len):
    import torch as _torch
    import torch.nn as _nn
    import numpy as _np
    import plotly.graph_objects as _go

    seq_len = vanish_seq_len.value
    hidden_size = 16
    input_size = 4

    # Create model
    if vanish_model_type.value == "Vanilla RNN":
        model = _nn.RNN(input_size, hidden_size, batch_first=True)
    else:
        model = _nn.LSTM(input_size, hidden_size, batch_first=True)

    # Random input
    _x = _torch.randn(1, seq_len, input_size, requires_grad=True)

    # Forward pass
    output, _ = model(_x)

    # Compute gradient of last output w.r.t. input at each time step
    grad_norms = []
    _target = output[0, -1, :].sum()  # scalar loss from last time step
    _target.backward()

    # Gradient of loss w.r.t. input at each time step
    input_grad = _x.grad[0]  # (seq_len, input_size)
    for _t in range(seq_len):
        grad_norms.append(input_grad[_t].norm().item())

    _fig = _go.Figure()
    _fig.add_trace(
        _go.Bar(
            x=list(range(seq_len)),
            y=grad_norms,
            marker_color=["#EF553B" if g < 0.01 else "#636EFA" for g in grad_norms],
        )
    )

    _fig.update_layout(
        template="plotly_dark",
        title=f"{vanish_model_type.value} | Gradient magnitude by time step (seq_len={seq_len})",
        xaxis_title="Time Step (0 = earliest)",
        yaxis_title="Gradient Norm",
        yaxis_type="log",
        height=350,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. LSTM: Long Short-Term Memory

    LSTMs solve vanishing gradients by adding a **cell state** $C_t$ — a "highway" that
    allows information to flow across many time steps with minimal degradation.

    Three gates control information flow:

    | Gate | Formula | Purpose |
    |------|---------|---------|
    | **Forget** | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ | What to **discard** from cell state |
    | **Input** | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ | What **new info** to store |
    | **Output** | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ | What to **output** from cell state |

    Cell state update:
    $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
    $$h_t = o_t \odot \tanh(C_t)$$

    The key: $C_t$ can pass through many steps if the forget gate stays close to 1.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### LSTM Gate Visualizer

    Adjust the gate values below to see how they control information flow.
    Think of the cell state as a "memory bank" and the gates as access controls:
    """)
    return


@app.cell
def _(mo):
    forget_gate = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.9, label="Forget Gate (f)")
    input_gate = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.5, label="Input Gate (i)")
    output_gate = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.8, label="Output Gate (o)")
    mo.hstack([forget_gate, input_gate, output_gate], justify="center", gap=1)
    return forget_gate, input_gate, output_gate


@app.cell
def _(forget_gate, input_gate, mo, output_gate):
    import numpy as _np

    _forget = forget_gate.value
    _input = input_gate.value
    _output = output_gate.value

    # Simulate with example values
    C_prev = 0.7  # Previous cell state
    C_candidate = 0.4  # New candidate value

    C_new = _forget * C_prev + _input * C_candidate
    h_new = _output * _np.tanh(C_new)

    # Build visual explanation
    bar_width = 30

    def bar(val, max_val=1.0, color="blue"):
        filled = int(val / max_val * bar_width)
        return f"`{'█' * filled}{'░' * (bar_width - filled)}` {val:.2f}"

    mo.md(
        f"""
        ### Information Flow

        **Previous cell state** ($C_{{t-1}}$): {bar(C_prev, color="blue")}

        **Step 1: Forget** — multiply by forget gate ({_forget:.2f}):
        - Kept from memory: {bar(_forget * C_prev)}
        - {'✅ Remembering most of the past' if _forget > 0.7 else '⚠️ Forgetting a lot!' if _forget < 0.3 else '🔄 Partially remembering'}

        **Step 2: Input** — add new info scaled by input gate ({_input:.2f}):
        - New candidate: {C_candidate:.2f} × {_input:.2f} = {_input * C_candidate:.3f}
        - {'✅ Storing new information' if _input > 0.7 else '⚠️ Ignoring new input' if _input < 0.3 else '🔄 Partially storing'}

        **New cell state** ($C_t$): {bar(C_new, max_val=1.5)}
        $$C_t = {_forget:.2f} \\times {C_prev:.2f} + {_input:.2f} \\times {C_candidate:.2f} = {C_new:.3f}$$

        **Step 3: Output** — scale by output gate ({_output:.2f}):
        - Hidden state: $h_t = {_output:.2f} \\times \\tanh({C_new:.3f}) = {h_new:.3f}$
        - {'✅ Exposing cell state to next layer' if _output > 0.7 else '⚠️ Hiding cell state' if _output < 0.3 else '🔄 Partially exposing'}

        ---
        **Intuition:**
        - **Forget gate ≈ 1:** "Keep remembering" (good for long-range dependencies)
        - **Input gate ≈ 0:** "Nothing new to store" (current input is not important)
        - **Output gate ≈ 0:** "Don't reveal what I know yet" (cell stores info silently)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Character-Level Text Generation with LSTM

    Train an LSTM to predict the next character in a sequence, then use it to
    generate new text. The model learns character patterns from the training text:
    """)
    return


@app.cell
def _(mo):
    gen_train_btn = mo.ui.run_button(label="Train & Generate")
    gen_temperature = mo.ui.slider(
        start=0.1, stop=2.0, step=0.1, value=0.8, label="Temperature"
    )
    gen_length = mo.ui.slider(
        start=50, stop=300, step=50, value=150, label="Generated Length"
    )
    mo.hstack([gen_train_btn, gen_temperature, gen_length], justify="center", gap=1)
    return gen_length, gen_temperature, gen_train_btn


@app.cell
def _(gen_length, gen_temperature, gen_train_btn, mo):
    import torch as _torch
    import torch.nn as _nn

    gen_train_btn.value

    # Training text
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
    For in that sleep of death what dreams may come"""

    _chars = sorted(set(corpus))
    _char_to_idx = {c: i for i, c in enumerate(_chars)}
    _idx_to_char = {i: c for c, i in _char_to_idx.items()}
    _vocab_size = len(_chars)

    # Prepare sequences
    seq_length = 20
    sequences, targets = [], []
    for _i in range(len(corpus) - seq_length):
        seq = [_char_to_idx[c] for c in corpus[_i:_i+seq_length]]
        _target_idx = _char_to_idx[corpus[_i+seq_length]]
        sequences.append(seq)
        targets.append(_target_idx)

    X = _torch.LongTensor(sequences)
    Y = _torch.LongTensor(targets)

    # Model
    class CharLSTM(_nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim):
            super().__init__()
            self.embed = _nn.Embedding(vocab_size, embed_dim)
            self.lstm = _nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = _nn.Linear(hidden_dim, vocab_size)

        def forward(self, x, hidden=None):
            emb = self.embed(x)
            out, hidden = self.lstm(emb, hidden)
            logits = self.fc(out[:, -1, :])
            return logits, hidden

    char_model = CharLSTM(_vocab_size, embed_dim=32, hidden_dim=64)
    optimizer = _torch.optim.Adam(char_model.parameters(), lr=0.005)
    criterion = _nn.CrossEntropyLoss()

    # Train
    losses = []
    for epoch in range(100):
        # Mini-batch
        _batch_idx = _torch.randint(0, len(X), (64,))
        xb, yb = X[_batch_idx], Y[_batch_idx]
        logits, _ = char_model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Generate text
    char_model.eval()
    seed = corpus[:seq_length]
    generated = seed
    input_seq = _torch.LongTensor([[_char_to_idx[c] for c in seed]])
    hidden = None

    with _torch.no_grad():
        for _ in range(gen_length.value):
            logits, hidden = char_model(input_seq, hidden)
            # Temperature sampling
            probs = _torch.softmax(logits / gen_temperature.value, dim=-1)
            next_idx = _torch.multinomial(probs, 1).item()
            generated += _idx_to_char[next_idx]
            input_seq = _torch.LongTensor([[next_idx]])

    mo.md(
        f"""
        ### Generated Text (temperature={gen_temperature.value})

        ```
        {generated}
        ```

        **Temperature controls randomness:**
        - **Low (0.1-0.5):** Conservative, repetitive but "safe" text
        - **Medium (0.7-1.0):** Balanced creativity and coherence
        - **High (1.5-2.0):** Wild and creative, more errors

        Final training loss: {losses[-1]:.4f}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | What You Learned |
    |---------|-----------------|
    | **RNN** | Processes sequences by maintaining a hidden state that evolves over time |
    | **Hidden state** | Compressed summary of everything seen so far |
    | **Vanishing gradients** | Gradients decay exponentially over long sequences |
    | **LSTM** | Solves vanishing gradients with cell state + gates (forget, input, output) |
    | **Temperature** | Controls randomness in text generation |

    ### Key Takeaway
    RNNs/LSTMs brought **memory** to neural networks — the ability to process sequences
    where order matters. But they process tokens **one at a time** (sequentially), which
    is slow and makes it hard to capture very long-range dependencies.

    **Next up:** [Notebook 05 — Attention Mechanism](05_attention_mechanism.py) —
    a revolutionary idea that lets models look at the *entire* sequence at once.
    """)
    return


if __name__ == "__main__":
    app.run()
