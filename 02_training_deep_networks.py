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
    # 02 — Training Deep Networks

    ## Learning Objectives
    - Understand loss functions and why we need them
    - Visualize gradient descent on a loss surface
    - Step through backpropagation (the chain rule in action)
    - Compare optimizers: SGD, Momentum, Adam
    - Diagnose overfitting and apply regularization (dropout, batch norm)

    ## Connection to Classical ML
    You know **MSE** (Mean Squared Error) and **R²** from regression, and **cross-entropy** from
    logistic regression. Neural networks use the *same* loss functions — MSE for regression tasks,
    cross-entropy for classification. The difference is *how* we minimize them: instead of a
    closed-form solution (like OLS), we use **iterative gradient descent**.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Loss Functions: Measuring How Wrong We Are

    A loss function $\mathcal{L}(\hat{y}, y)$ quantifies the gap between predictions and truth.

    | Loss Function | Formula | Use Case |
    |--------------|---------|----------|
    | **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Regression |
    | **Binary Cross-Entropy** | $-\frac{1}{n}\sum[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$ | Binary classification |
    | **Cross-Entropy** | $-\frac{1}{n}\sum\sum y_{ic} \log \hat{y}_{ic}$ | Multi-class classification |

    Cross-entropy penalizes *confident wrong predictions* heavily — predicting 0.01
    when the true label is 1 costs much more than predicting 0.4.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Gradient Descent Visualizer

    Gradient descent finds the minimum of the loss function by repeatedly:
    1. Compute the gradient $\nabla \mathcal{L}$ (which direction is "downhill")
    2. Take a step: $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$

    The **learning rate** $\eta$ controls step size:
    - Too large → overshoots the minimum, may diverge
    - Too small → converges very slowly
    - Just right → smooth convergence

    Try adjusting the learning rate and optimizer below to see the effect on the optimization path:
    """)
    return


@app.cell
def _(mo):
    gd_lr_slider = mo.ui.slider(
        start=0.001, stop=0.5, step=0.001, value=0.05, label="Learning Rate"
    )
    gd_optimizer_dropdown = mo.ui.dropdown(
        options=["SGD", "SGD + Momentum", "Adam"],
        value="SGD",
        label="Optimizer",
    )
    gd_steps_slider = mo.ui.slider(
        start=10, stop=200, step=10, value=50, label="Steps"
    )
    gd_run_button = mo.ui.run_button(label="Run Optimization")
    mo.hstack(
        [gd_lr_slider, gd_optimizer_dropdown, gd_steps_slider, gd_run_button],
        justify="center",
        gap=1,
    )
    return gd_lr_slider, gd_optimizer_dropdown, gd_run_button, gd_steps_slider


@app.cell
def _(gd_lr_slider, gd_optimizer_dropdown, gd_run_button, gd_steps_slider, mo):
    import numpy as _np
    import plotly.graph_objects as _go

    gd_run_button.value

    # Rosenbrock-like loss surface: f(x,y) = (1-x)^2 + 10*(y-x^2)^2
    def loss_surface(x, y):
        return (1 - x) ** 2 + 10 * (y - x ** 2) ** 2

    def grad_loss(x, y):
        dx = -2 * (1 - x) - 40 * x * (y - x ** 2)
        dy = 20 * (y - x ** 2)
        return _np.array([dx, dy])

    # Optimization path
    lr = gd_lr_slider.value
    pos = _np.array([-1.5, 2.0])
    velocity = _np.array([0.0, 0.0])
    m = _np.array([0.0, 0.0])  # Adam first moment
    v_adam = _np.array([0.0, 0.0])  # Adam second moment
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    path_x, path_y, path_z = [pos[0]], [pos[1]], [loss_surface(pos[0], pos[1])]

    for t in range(1, gd_steps_slider.value + 1):
        g = grad_loss(pos[0], pos[1])
        if gd_optimizer_dropdown.value == "SGD":
            pos = pos - lr * g
        elif gd_optimizer_dropdown.value == "SGD + Momentum":
            velocity = 0.9 * velocity + g
            pos = pos - lr * velocity
        else:  # Adam
            m = beta1 * m + (1 - beta1) * g
            v_adam = beta2 * v_adam + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v_adam / (1 - beta2 ** t)
            pos = pos - lr * m_hat / (_np.sqrt(v_hat) + eps)

        pos = _np.clip(pos, -3, 3)
        path_x.append(pos[0])
        path_y.append(pos[1])
        path_z.append(loss_surface(pos[0], pos[1]))

    # Plot
    xx, yy = _np.meshgrid(_np.linspace(-2, 2, 80), _np.linspace(-1, 3, 80))
    zz = loss_surface(xx, yy)

    _fig = _go.Figure()
    _fig.add_trace(
        _go.Contour(
            x=_np.linspace(-2, 2, 80), y=_np.linspace(-1, 3, 80), z=zz,
            colorscale="Viridis", showscale=False,
            contours=dict(start=0, end=50, size=2),
            opacity=0.7,
        )
    )
    _fig.add_trace(
        _go.Scatter(
            x=path_x, y=path_y, mode="lines+markers",
            line=dict(color="#EF553B", width=2),
            marker=dict(size=4, color="#EF553B"),
            name="Optimizer path",
        )
    )
    _fig.add_trace(
        _go.Scatter(
            x=[path_x[0]], y=[path_y[0]], mode="markers",
            marker=dict(size=12, color="#FFA15A", symbol="star"),
            name="Start",
        )
    )
    _fig.add_trace(
        _go.Scatter(
            x=[1], y=[1], mode="markers",
            marker=dict(size=12, color="#00CC96", symbol="star"),
            name="Minimum (1,1)",
        )
    )

    _fig.update_layout(
        template="plotly_dark",
        title=f"{gd_optimizer_dropdown.value} | LR={lr} | Final loss: {path_z[-1]:.4f}",
        xaxis_title="θ₁", yaxis_title="θ₂",
        height=450,
        margin=dict(l=50, r=30, t=60, b=50),
    )

    mo.md(
        f"""
        **Optimizer comparison tips:**
        - **SGD** with high LR (>0.1): watch it oscillate or diverge
        - **SGD + Momentum**: smoother path, builds up speed in consistent directions
        - **Adam**: adaptive per-parameter LR — usually converges fastest

        Final position: ({path_x[-1]:.3f}, {path_y[-1]:.3f}) | Target: (1.0, 1.0)
        """
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Backpropagation: The Chain Rule in Action

    How does the network compute gradients? **Backpropagation** = applying the
    chain rule of calculus, layer by layer, from output back to input.

    For a simple 2-layer network $\hat{y} = \sigma(w_2 \cdot \text{ReLU}(w_1 x + b_1) + b_2)$:

    **Forward pass** (compute output):
    $$z_1 = w_1 x + b_1 \quad \rightarrow \quad h = \text{ReLU}(z_1) \quad \rightarrow \quad z_2 = w_2 h + b_2 \quad \rightarrow \quad \hat{y} = \sigma(z_2)$$

    **Backward pass** (compute gradients via chain rule):
    $$\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} \cdot \frac{\partial z_2}{\partial w_2}$$

    $$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} \cdot \frac{\partial z_2}{\partial h} \cdot \frac{\partial h}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

    Each factor is simple on its own. The chain rule *chains* them together.
    PyTorch computes all of this automatically with `loss.backward()`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step-Through: Backprop on a Tiny Network

    Let's trace forward and backward passes through a concrete example.
    Use the slider to step through each computation:
    """)
    return


@app.cell
def _(mo):
    backprop_step = mo.ui.slider(
        start=0, stop=7, step=1, value=0, label="Step"
    )
    backprop_step
    return (backprop_step,)


@app.cell
def _(backprop_step, mo):
    import numpy as _np

    # Fixed example values
    x_bp = 1.5
    w1, b1 = 0.8, -0.5
    w2, b2 = 1.2, 0.3
    y_true = 1.0

    # Forward pass computations
    z1 = w1 * x_bp + b1  # 0.7
    h = max(0, z1)  # ReLU: 0.7
    z2 = w2 * h + b2  # 1.14
    y_hat = 1 / (1 + _np.exp(-z2))  # sigmoid: ~0.757
    loss_value = -(y_true * _np.log(y_hat) + (1 - y_true) * _np.log(1 - y_hat))

    # Backward pass computations
    dL_dyhat = -y_true / y_hat + (1 - y_true) / (1 - y_hat)
    dyhat_dz2 = y_hat * (1 - y_hat)
    dL_dz2 = dL_dyhat * dyhat_dz2
    dL_dw2 = dL_dz2 * h
    dL_db2 = dL_dz2
    dz2_dh = w2
    dh_dz1 = 1.0 if z1 > 0 else 0.0
    dL_dw1 = dL_dz2 * dz2_dh * dh_dz1 * x_bp
    dL_db1 = dL_dz2 * dz2_dh * dh_dz1

    steps = [
        f"""
        **Step 0: Setup**
        - Input: x = {x_bp}
        - Weights: w₁ = {w1}, b₁ = {b1}, w₂ = {w2}, b₂ = {b2}
        - True label: y = {y_true}
        """,
        f"""
        **Step 1: Forward — Linear Layer 1**
        $$z_1 = w_1 \\cdot x + b_1 = {w1} \\times {x_bp} + ({b1}) = {z1:.2f}$$
        """,
        f"""
        **Step 2: Forward — ReLU Activation**
        $$h = \\text{{ReLU}}(z_1) = \\max(0, {z1:.2f}) = {h:.2f}$$
        ReLU passes positive values through unchanged, zeros out negatives.
        """,
        f"""
        **Step 3: Forward — Linear Layer 2**
        $$z_2 = w_2 \\cdot h + b_2 = {w2} \\times {h:.2f} + {b2} = {z2:.2f}$$
        """,
        f"""
        **Step 4: Forward — Sigmoid + Loss**
        $$\\hat{{y}} = \\sigma(z_2) = \\frac{{1}}{{1+e^{{-{z2:.2f}}}}} = {y_hat:.4f}$$
        $$\\mathcal{{L}} = -[y\\log\\hat{{y}} + (1-y)\\log(1-\\hat{{y}})] = {loss_value:.4f}$$
        """,
        f"""
        **Step 5: Backward — Output Gradients**
        $$\\frac{{\\partial \\mathcal{{L}}}}{{\\partial z_2}} = \\hat{{y}} - y = {y_hat:.4f} - {y_true} = {dL_dz2:.4f}$$
        (This simplifies nicely for BCE + sigmoid!)

        $$\\frac{{\\partial \\mathcal{{L}}}}{{\\partial w_2}} = \\frac{{\\partial \\mathcal{{L}}}}{{\\partial z_2}} \\cdot h = {dL_dz2:.4f} \\times {h:.2f} = {dL_dw2:.4f}$$
        """,
        f"""
        **Step 6: Backward — Hidden Layer Gradients**
        $$\\frac{{\\partial \\mathcal{{L}}}}{{\\partial h}} = \\frac{{\\partial \\mathcal{{L}}}}{{\\partial z_2}} \\cdot w_2 = {dL_dz2:.4f} \\times {w2} = {dL_dz2 * dz2_dh:.4f}$$

        $$\\frac{{\\partial h}}{{\\partial z_1}} = \\begin{{cases}} 1 & \\text{{if }} z_1 > 0 \\\\ 0 & \\text{{otherwise}} \\end{{cases}} = {dh_dz1:.0f}$$
        (ReLU derivative: 1 for positive inputs, 0 otherwise)
        """,
        f"""
        **Step 7: Backward — Input Layer Gradients**
        $$\\frac{{\\partial \\mathcal{{L}}}}{{\\partial w_1}} = \\frac{{\\partial \\mathcal{{L}}}}{{\\partial z_2}} \\cdot w_2 \\cdot \\frac{{\\partial h}}{{\\partial z_1}} \\cdot x = {dL_dw1:.4f}$$
        $$\\frac{{\\partial \\mathcal{{L}}}}{{\\partial b_1}} = {dL_db1:.4f}$$

        **Weight updates** (with LR = 0.1):
        - $w_1 \\leftarrow {w1} - 0.1 \\times {dL_dw1:.4f} = {w1 - 0.1 * dL_dw1:.4f}$
        - $w_2 \\leftarrow {w2} - 0.1 \\times {dL_dw2:.4f} = {w2 - 0.1 * dL_dw2:.4f}$
        """,
    ]

    step = backprop_step.value
    is_forward = step <= 4
    phase = "FORWARD PASS" if is_forward else "BACKWARD PASS"
    color = "#636EFA" if is_forward else "#EF553B"

    mo.md(
        f"""
        <div style="border-left: 4px solid {color}; padding-left: 16px;">

        **Phase: {phase}** (Step {step}/7)

        {steps[step]}

        </div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Training Dashboard

    Train a network on MNIST (handwritten digits) with adjustable hyperparameters.
    Watch how learning rate, batch size, and dropout affect training:
    """)
    return


@app.cell
def _(mo):
    dash_lr = mo.ui.slider(
        start=0.0001, stop=0.01, step=0.0001, value=0.001, label="Learning Rate"
    )
    dash_batch = mo.ui.dropdown(
        options={"32": 32, "64": 64, "128": 128, "256": 256},
        value="64",
        label="Batch Size",
    )
    dash_dropout = mo.ui.slider(
        start=0.0, stop=0.5, step=0.05, value=0.0, label="Dropout Rate"
    )
    dash_epochs = mo.ui.slider(
        start=3, stop=15, step=1, value=5, label="Epochs"
    )
    dash_train_btn = mo.ui.run_button(label="Train on MNIST")
    mo.hstack(
        [dash_lr, dash_batch, dash_dropout, dash_epochs, dash_train_btn],
        justify="center",
        gap=1,
    )
    return dash_batch, dash_dropout, dash_epochs, dash_lr, dash_train_btn


@app.cell
def _(dash_batch, dash_dropout, dash_epochs, dash_lr, dash_train_btn, mo):
    import torch
    import torch.nn as _nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import fetch_openml
    import numpy as _np
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    dash_train_btn.value

    mo.status.spinner(title="Loading MNIST...", remove_on_exit=True)

    # Load a subset of MNIST for speed
    from sklearn.datasets import load_digits
    digits = load_digits()
    X_all = digits.data.astype(_np.float32) / 16.0  # normalize to [0,1]
    y_all = digits.target.astype(_np.int64)

    # Train/val split
    n_train = int(0.8 * len(X_all))
    X_train, X_val = X_all[:n_train], X_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    batch_size = int(dash_batch.value)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    dropout_rate = dash_dropout.value
    input_dim = X_train.shape[1]  # 64 for sklearn digits

    mnist_model = _nn.Sequential(
        _nn.Linear(input_dim, 128),
        _nn.ReLU(),
        _nn.Dropout(dropout_rate),
        _nn.Linear(128, 64),
        _nn.ReLU(),
        _nn.Dropout(dropout_rate),
        _nn.Linear(64, 10),
    )

    optimizer = torch.optim.Adam(mnist_model.parameters(), lr=dash_lr.value)
    criterion = _nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(dash_epochs.value):
        # Training
        mnist_model.train()
        epoch_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            pred = mnist_model(xb)
            _loss = criterion(pred, yb)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            epoch_loss += _loss.item() * len(xb)
            correct += (pred.argmax(1) == yb).sum().item()
            total += len(xb)
        train_losses.append(epoch_loss / total)
        train_accs.append(correct / total)

        # Validation
        mnist_model.eval()
        epoch_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = mnist_model(xb)
                _loss = criterion(pred, yb)
                epoch_loss += _loss.item() * len(xb)
                correct += (pred.argmax(1) == yb).sum().item()
                total += len(xb)
        val_losses.append(epoch_loss / total)
        val_accs.append(correct / total)

    # Plot
    _fig = _make_subplots(rows=1, cols=2, subplot_titles=["Loss", "Accuracy"])
    epochs_range = list(range(1, len(train_losses) + 1))

    _fig.add_trace(_go.Scatter(x=epochs_range, y=train_losses, name="Train Loss", line=dict(color="#636EFA")), row=1, col=1)
    _fig.add_trace(_go.Scatter(x=epochs_range, y=val_losses, name="Val Loss", line=dict(color="#EF553B")), row=1, col=1)
    _fig.add_trace(_go.Scatter(x=epochs_range, y=train_accs, name="Train Acc", line=dict(color="#636EFA")), row=1, col=2)
    _fig.add_trace(_go.Scatter(x=epochs_range, y=val_accs, name="Val Acc", line=dict(color="#EF553B")), row=1, col=2)

    _fig.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=50, r=30, t=60, b=50),
        title=f"MNIST Training | LR={dash_lr.value} | Batch={batch_size} | Dropout={dropout_rate}",
    )
    _fig.update_xaxes(title_text="Epoch")

    gap = train_accs[-1] - val_accs[-1]
    overfit_msg = ""
    if gap > 0.05:
        overfit_msg = f"**Warning:** Train-Val accuracy gap is {gap:.1%} — signs of overfitting. Try increasing dropout."
    elif gap < 0.01:
        overfit_msg = "Good generalization — train and val accuracy are close."

    mo.md(
        f"""
        **Final Results:**
        | Train Acc: {train_accs[-1]:.1%} | Val Acc: {val_accs[-1]:.1%} | Gap: {gap:.1%}

        {overfit_msg}

        **Experiments to try:**
        - Set dropout to 0, train for 15 epochs → watch train/val diverge (overfitting)
        - Very high LR (0.01) → unstable training
        - Very low LR (0.0001) → slow convergence
        """
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Overfitting Demo

    Overfitting = the model memorizes training data instead of learning generalizable patterns.
    You know this from your bias-variance tradeoff notes. Here's a visual:
    """)
    return


@app.cell
def _(mo):
    overfit_complexity = mo.ui.slider(
        start=1, stop=20, step=1, value=3, label="Model Complexity (polynomial degree)"
    )
    overfit_noise = mo.ui.slider(
        start=0.0, stop=2.0, step=0.1, value=0.5, label="Noise Level"
    )
    mo.hstack([overfit_complexity, overfit_noise], justify="center", gap=1.5)
    return overfit_complexity, overfit_noise


@app.cell
def _(overfit_complexity, overfit_noise):
    import numpy as _np
    import plotly.graph_objects as _go

    _np.random.seed(42)
    n_points = 30
    x_data = _np.linspace(0, 1, n_points)
    y_true_fn = _np.sin(2 * _np.pi * x_data)
    y_noisy = y_true_fn + overfit_noise.value * _np.random.randn(n_points)

    # Fit polynomial
    degree = overfit_complexity.value
    coeffs = _np.polyfit(x_data, y_noisy, degree)
    x_smooth = _np.linspace(0, 1, 200)
    y_fit = _np.polyval(coeffs, x_smooth)

    # Train/test error
    y_train_pred = _np.polyval(coeffs, x_data)
    train_mse = _np.mean((y_noisy - y_train_pred) ** 2)

    # Test on new points
    x_test = _np.random.uniform(0, 1, 50)
    y_test_true = _np.sin(2 * _np.pi * x_test) + overfit_noise.value * _np.random.randn(50)
    y_test_pred = _np.polyval(coeffs, x_test)
    test_mse = _np.mean((y_test_true - y_test_pred) ** 2)

    _fig = _go.Figure()
    _fig.add_trace(_go.Scatter(x=x_smooth, y=_np.sin(2 * _np.pi * x_smooth), name="True function", line=dict(color="#00CC96", dash="dash")))
    _fig.add_trace(_go.Scatter(x=x_data, y=y_noisy, mode="markers", name="Training data", marker=dict(color="#636EFA", size=6)))
    _fig.add_trace(_go.Scatter(x=x_smooth, y=y_fit, name=f"Degree {degree} fit", line=dict(color="#EF553B", width=2)))

    _fig.update_layout(
        template="plotly_dark",
        title=f"Polynomial Degree {degree} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}",
        xaxis_title="x", yaxis_title="y",
        yaxis=dict(range=[-3, 3]),
        height=380,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | What You Learned |
    |---------|-----------------|
    | **Loss functions** | MSE for regression, cross-entropy for classification — same as classical ML |
    | **Gradient descent** | Iteratively step in the direction of steepest descent |
    | **Backpropagation** | Chain rule applied layer-by-layer to compute all gradients |
    | **Optimizers** | SGD → Momentum → Adam: each adds intelligence to the update rule |
    | **Overfitting** | Train/val gap signals overfitting; dropout and early stopping help |

    ### Key Takeaway
    Training a neural network is an optimization problem: find the weights that minimize
    the loss function. Backpropagation computes the gradients, and the optimizer decides
    how to use them. The art is in choosing the right hyperparameters.

    **Next up:** [Notebook 03 — Convolutional Networks](03_convolutional_networks.py) —
    how do we process images and learn spatial features?
    """)
    return


if __name__ == "__main__":
    app.run()
