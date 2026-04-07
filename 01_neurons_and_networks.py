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
    # 01 — Neurons & Networks

    ## Learning Objectives
    - Understand how a single artificial neuron (perceptron) works
    - Compare different activation functions and see how they shape outputs
    - Build multi-layer networks and visualize forward passes
    - Train a simple classifier and watch decision boundaries form

    ## Connection to Classical ML
    You already know **logistic regression**: take a linear combination of inputs $z = w^T x + b$,
    pass it through a **sigmoid** $\sigma(z) = \frac{1}{1+e^{-z}}$, and get a probability.

    **A single neuron does exactly the same thing** — but we can choose different activation
    functions, and we can *stack* many neurons into layers. That's the leap from logistic
    regression to neural networks.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The Single Neuron (Perceptron)

    A neuron computes:

    $$y = f(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b) = f(\mathbf{w}^T \mathbf{x} + b)$$

    where:
    - $\mathbf{x}$ = input features (like your regression predictors)
    - $\mathbf{w}$ = weights (like regression coefficients)
    - $b$ = bias (like the intercept)
    - $f$ = **activation function** (sigmoid in logistic regression, but we have more choices now)

    The activation function is what makes neurons non-linear. Without it, stacking layers
    would just be matrix multiplication — equivalent to a single linear model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Activation Function Explorer

    Adjust the weight and bias below to see how different activation functions
    transform the input. Notice:
    - **Sigmoid**: squashes to (0, 1) — great for probabilities
    - **Tanh**: squashes to (-1, 1) — zero-centered, often better for hidden layers
    - **ReLU**: simple thresholding — most popular in modern networks
    - **Leaky ReLU**: fixes the "dying neuron" problem where ReLU outputs zero for negative inputs
    """)
    return


@app.cell
def _(mo):
    weight_slider = mo.ui.slider(
        start=-3.0, stop=3.0, step=0.1, value=1.0, label="Weight (w)"
    )
    bias_slider = mo.ui.slider(
        start=-3.0, stop=3.0, step=0.1, value=0.0, label="Bias (b)"
    )
    activation_dropdown = mo.ui.dropdown(
        options=["Sigmoid", "Tanh", "ReLU", "Leaky ReLU"],
        value="Sigmoid",
        label="Activation Function",
    )
    mo.hstack(
        [weight_slider, bias_slider, activation_dropdown],
        justify="center",
        gap=1.5,
    )
    return activation_dropdown, bias_slider, weight_slider


@app.cell
def _(activation_dropdown, bias_slider, weight_slider):
    import numpy as _np
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    x_range = _np.linspace(-5, 5, 200)
    w = weight_slider.value
    b = bias_slider.value
    z = w * x_range + b

    def sigmoid(x):
        return 1 / (1 + _np.exp(-_np.clip(x, -500, 500)))

    def tanh(x):
        return _np.tanh(x)

    def relu(x):
        return _np.maximum(0, x)

    def leaky_relu(x):
        return _np.where(x > 0, x, 0.01 * x)

    activation_fns = {
        "Sigmoid": sigmoid,
        "Tanh": tanh,
        "ReLU": relu,
        "Leaky ReLU": leaky_relu,
    }

    activation_formulas = {
        "Sigmoid": "σ(z) = 1 / (1 + e⁻ᶻ)",
        "Tanh": "tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)",
        "ReLU": "ReLU(z) = max(0, z)",
        "Leaky ReLU": "LeakyReLU(z) = max(0.01z, z)",
    }

    fn = activation_fns[activation_dropdown.value]
    activated_output = fn(z)

    fig = _make_subplots(
        rows=1, cols=2,
        subplot_titles=["Linear transformation: z = wx + b", f"After activation: {activation_dropdown.value}"],
    )

    fig.add_trace(
        _go.Scatter(x=x_range, y=z, name="z = wx + b", line=dict(color="#636EFA", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        _go.Scatter(
            x=x_range,
            y=activated_output,
            name=f"{activation_dropdown.value}(z)",
            line=dict(color="#EF553B", width=2),
        ),
        row=1, col=2,
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=30, t=60, b=50),
        title=f"Neuron: y = {activation_dropdown.value}({w:.1f}·x + {b:.1f})    Formula: {activation_formulas[activation_dropdown.value]}",
        showlegend=False,
    )
    fig.update_xaxes(title_text="x (input)")
    fig.update_yaxes(title_text="z", row=1, col=1)
    fig.update_yaxes(title_text="y (output)", row=1, col=2)
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. From One Neuron to a Network

    A **multi-layer network** (MLP) stacks neurons into layers:

    $$\mathbf{h}_1 = f_1(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
    $$\mathbf{h}_2 = f_2(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$
    $$\hat{y} = f_{out}(\mathbf{W}_{out} \mathbf{h}_2 + \mathbf{b}_{out})$$

    Each layer applies a **linear transformation** (matrix multiply + bias) followed by a
    **non-linear activation**. The key insight: **depth creates composition**, allowing the
    network to learn hierarchical features.

    | Layer | Analogy from Classical ML |
    |-------|--------------------------|
    | Input layer | Your feature matrix $X$ |
    | Hidden layers | Automatic feature engineering (like polynomial features, but learned) |
    | Output layer | The final prediction (sigmoid for binary, softmax for multi-class) |

    ### Network Architecture Visualizer

    Adjust the sliders to build different architectures and see the network structure:
    """)
    return


@app.cell
def _(mo):
    n_hidden_layers = mo.ui.slider(
        start=1, stop=4, step=1, value=2, label="Hidden Layers"
    )
    neurons_per_layer = mo.ui.slider(
        start=2, stop=16, step=1, value=8, label="Neurons per Hidden Layer"
    )
    mo.hstack([n_hidden_layers, neurons_per_layer], justify="center", gap=1.5)
    return n_hidden_layers, neurons_per_layer


@app.cell
def _(n_hidden_layers, neurons_per_layer):
    import numpy as _np
    import plotly.graph_objects as _go

    def _plot_network(layers: list[int], title: str) -> _go.Figure:
        fig = _go.Figure()
        max_neurons = max(layers)
        x_spacing = 1.5
        node_x, node_y, node_text = [], [], []

        for layer_idx, n_neurons in enumerate(layers):
            x = layer_idx * x_spacing
            y_positions = _np.linspace(
                -(n_neurons - 1) / 2, (n_neurons - 1) / 2, n_neurons
            )

            for neuron_idx, y in enumerate(y_positions):
                node_x.append(x)
                node_y.append(y)
                if layer_idx == 0:
                    node_text.append(f"Input {neuron_idx + 1}")
                elif layer_idx == len(layers) - 1:
                    node_text.append(f"Output {neuron_idx + 1}")
                else:
                    node_text.append(f"L{layer_idx} N{neuron_idx + 1}")

            if layer_idx > 0:
                prev_n = layers[layer_idx - 1]
                prev_x = (layer_idx - 1) * x_spacing
                prev_y_positions = _np.linspace(
                    -(prev_n - 1) / 2, (prev_n - 1) / 2, prev_n
                )
                for prev_y in prev_y_positions:
                    for current_y in y_positions:
                        fig.add_trace(
                            _go.Scatter(
                                x=[prev_x, x],
                                y=[prev_y, current_y],
                                mode="lines",
                                line=dict(color="rgba(150,150,150,0.3)", width=1),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

        fig.add_trace(
            _go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=25,
                    color="#636EFA",
                    line=dict(width=2, color="white"),
                ),
                text=node_text,
                textposition="top center",
                textfont=dict(size=8),
                showlegend=False,
            )
        )

        for layer_idx, n_neurons in enumerate(layers):
            label = (
                "Input"
                if layer_idx == 0
                else ("Output" if layer_idx == len(layers) - 1 else f"Hidden {layer_idx}")
            )
            fig.add_annotation(
                x=layer_idx * x_spacing,
                y=max_neurons / 2 + 0.8,
                text=f"{label}<br>({n_neurons} neurons)",
                showarrow=False,
                font=dict(size=10),
            )

        fig.update_layout(
            template="plotly_dark",
            title=title,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x"),
            height=400,
            margin=dict(l=50, r=30, t=50, b=50),
            font=dict(size=12),
        )
        return fig

    layers = [2] + [neurons_per_layer.value] * n_hidden_layers.value + [1]
    total_params = sum(
        layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1)
    )
    fig_net = _plot_network(
        layers, title=f"Network: {layers}  |  Total parameters: {total_params}"
    )
    fig_net
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Forward Pass: Data Flowing Through the Network

    Let's trace a single data point through a network. Each layer:
    1. Multiplies the input by weights (matrix multiply)
    2. Adds bias
    3. Applies activation function

    The code below shows this in PyTorch — notice how `nn.Sequential` chains layers
    together, exactly matching the $\mathbf{h} = f(\mathbf{W}\mathbf{x} + \mathbf{b})$ formula.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ```python
    import torch
    import torch.nn as nn

    # A 3-layer network: 2 inputs → 8 hidden → 4 hidden → 1 output
    model = nn.Sequential(
        nn.Linear(2, 8),   # W₁: (2×8), b₁: (8,)
        nn.ReLU(),         # Activation
        nn.Linear(8, 4),   # W₂: (8×4), b₂: (4,)
        nn.ReLU(),         # Activation
        nn.Linear(4, 1),   # W₃: (4×1), b₃: (1,)
        nn.Sigmoid(),      # Output activation for binary classification
    )

    # Forward pass: data flows through each layer in sequence
    x = torch.tensor([0.5, -0.3])  # One data point with 2 features
    output = model(x)               # Calls model.forward(x) internally
    ```

    This is what you saw in your `nn.Sequential` notes — but now you understand *why*
    each layer is there and what the math looks like.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Decision Boundary Visualizer

    Now let's see a neural network **learn**. We'll train on the **moons dataset**
    (two interleaving half-circles) — a classic non-linear classification problem
    that logistic regression can't solve well.

    Adjust the hidden layer size and watch:
    - More neurons → more flexible boundary → can capture complex shapes
    - Too few neurons → underfitting (can't separate the classes)
    - The boundary forms as the network learns to map 2D space to class probabilities
    """)
    return


@app.cell
def _(mo):
    hidden_size_slider = mo.ui.slider(
        start=2, stop=32, step=2, value=8, label="Hidden Layer Size"
    )
    epochs_slider = mo.ui.slider(
        start=50, stop=500, step=50, value=200, label="Training Epochs"
    )
    lr_slider = mo.ui.slider(
        start=0.001, stop=0.1, step=0.001, value=0.01, label="Learning Rate"
    )
    train_button = mo.ui.run_button(label="Train Network")
    mo.hstack(
        [hidden_size_slider, epochs_slider, lr_slider, train_button],
        justify="center",
        gap=1,
    )
    return epochs_slider, hidden_size_slider, lr_slider, train_button


@app.cell
def _(epochs_slider, hidden_size_slider, lr_slider, mo, train_button):
    import numpy as _np
    import plotly.graph_objects as _go
    import torch
    import torch.nn as _nn
    from plotly.subplots import make_subplots as _make_subplots
    from sklearn.datasets import make_moons

    # Only run when button is clicked
    train_button.value

    # Generate data
    X_np, y_np = make_moons(n_samples=300, noise=0.2, random_state=42)
    X = torch.FloatTensor(X_np)
    y = torch.FloatTensor(y_np).unsqueeze(1)

    # Build model
    h = hidden_size_slider.value
    model = _nn.Sequential(
        _nn.Linear(2, h),
        _nn.ReLU(),
        _nn.Linear(h, h),
        _nn.ReLU(),
        _nn.Linear(h, 1),
        _nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_slider.value)
    loss_fn = _nn.BCELoss()

    # Train
    losses = []
    for epoch in range(epochs_slider.value):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Decision boundary
    xx, yy = _np.meshgrid(
        _np.linspace(X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5, 100),
        _np.linspace(X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5, 100),
    )
    grid = torch.FloatTensor(_np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        zz = model(grid).numpy().reshape(xx.shape)

    accuracy = ((model(X).detach() > 0.5).float() == y).float().mean().item()

    fig_1 = _make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Decision Boundary (Acc: {accuracy:.1%})",
            "Training Loss",
        ],
    )

    fig_1.add_trace(
        _go.Contour(
            x=xx[0], y=yy[:, 0], z=zz,
            colorscale="RdBu", opacity=0.6,
            showscale=False,
            contours=dict(start=0, end=1, size=0.1),
        ),
        row=1, col=1,
    )
    fig_1.add_trace(
        _go.Scatter(
            x=X_np[y_np == 0, 0], y=X_np[y_np == 0, 1],
            mode="markers", marker=dict(color="#EF553B", size=5),
            name="Class 0",
        ),
        row=1, col=1,
    )
    fig_1.add_trace(
        _go.Scatter(
            x=X_np[y_np == 1, 0], y=X_np[y_np == 1, 1],
            mode="markers", marker=dict(color="#636EFA", size=5),
            name="Class 1",
        ),
        row=1, col=1,
    )
    fig_1.add_trace(
        _go.Scatter(
            x=list(range(len(losses))), y=losses,
            name="Loss", line=dict(color="#00CC96"),
        ),
        row=1, col=2,
    )

    fig_1.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=60, b=50),
        title=f"2-Layer Network with {h} hidden neurons",
    )
    fig_1.update_xaxes(title_text="x₁", row=1, col=1)
    fig_1.update_yaxes(title_text="x₂", row=1, col=1)
    fig_1.update_xaxes(title_text="Epoch", row=1, col=2)
    fig_1.update_yaxes(title_text="BCE Loss", row=1, col=2)

    mo.md(
        f"""
        **Results:** Trained for {epochs_slider.value} epochs with LR={lr_slider.value}
        | Hidden neurons: {h} | Final loss: {losses[-1]:.4f} | Accuracy: {accuracy:.1%}

        Try:
        - **2 neurons**: Can it separate the moons? (Probably not well)
        - **8 neurons**: Should work nicely
        - **32 neurons**: Very flexible — might overfit with enough epochs
        """
    )
    fig_1
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Building It in PyTorch: Your First Neural Network

    Here's the complete code for a trainable classifier. This pattern —
    `nn.Module` subclass with `__init__` and `forward` — is how every
    PyTorch model is built:
    """)
    return


@app.cell
def _():
    import torch.nn as _nn

    class SimpleClassifier(_nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = _nn.Sequential(
                _nn.Linear(input_dim, hidden_dim),
                _nn.ReLU(),
                _nn.Linear(hidden_dim, hidden_dim),
                _nn.ReLU(),
                _nn.Linear(hidden_dim, output_dim),
                _nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    # Instantiate
    demo_model = SimpleClassifier(input_dim=2, hidden_dim=16, output_dim=1)

    # Count parameters
    n_params = sum(p.numel() for p in demo_model.parameters())
    print(f"Model architecture:\n{demo_model}")
    print(f"\nTotal trainable parameters: {n_params}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | What You Learned |
    |---------|-----------------|
    | **Neuron** | $y = f(\mathbf{w}^T\mathbf{x} + b)$ — same as logistic regression but generalized |
    | **Activation functions** | Sigmoid, Tanh, ReLU, Leaky ReLU — each with different properties |
    | **Multi-layer network** | Stack neurons into layers for automatic feature learning |
    | **Forward pass** | Data flows through layers: linear transform → activation → repeat |
    | **Decision boundary** | More neurons = more flexible boundaries, but risk of overfitting |

    ### Key Takeaway
    A neural network is just **many logistic regressions stacked together**, with non-linear
    activations between them. The "deep" in deep learning comes from having many layers,
    which lets the network learn hierarchical features automatically.

    **Next up:** [Notebook 02 — Training Deep Networks](02_training_deep_networks.py) —
    how does the network actually *learn* those weights? (Hint: backpropagation and gradient descent)
    """)
    return


if __name__ == "__main__":
    app.run()
