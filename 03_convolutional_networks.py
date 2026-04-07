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
    # 03 — Convolutional Networks (CNNs)

    ## Learning Objectives
    - Understand the convolution operation and why it's useful for spatial data
    - Visualize how filters/kernels detect features (edges, textures, patterns)
    - Build a CNN for image classification
    - See what learned filters actually look like

    ## Connection to Classical ML
    In classical ML, you engineer features manually — polynomial features, interaction terms,
    domain-specific transforms. CNNs **learn their own features** directly from pixel data.
    Each convolutional filter is a learned feature detector. The network discovers what
    features matter for the task, just like how Random Forest automatically selects
    important features, but at a much deeper level.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The Convolution Operation

    A 2D convolution slides a small **kernel** (filter) across the image, computing a
    weighted sum at each position:

    $$(I * K)[i,j] = \sum_{m}\sum_{n} I[i+m, j+n] \cdot K[m,n]$$

    Think of it as a "pattern matcher" — each kernel detects a specific local pattern
    (horizontal edge, vertical edge, corner, etc.).

    **Key parameters:**
    | Parameter | What it does |
    |-----------|-------------|
    | **Kernel size** | How large the local region is (typically 3x3 or 5x5) |
    | **Stride** | How many pixels to skip between positions (default: 1) |
    | **Padding** | Zero-padding around the image to control output size |

    **Output size formula:**
    $$\text{output} = \left\lfloor\frac{W - F + 2P}{S}\right\rfloor + 1$$

    where W = input width, F = filter size, P = padding, S = stride.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Convolution Step-Through

    Watch a 3x3 kernel slide across a small image, computing the output pixel by pixel.
    Use the slider to step through each position:
    """)
    return


@app.cell
def _(mo):
    conv_step = mo.ui.slider(
        start=0, stop=15, step=1, value=0, label="Convolution Step"
    )
    kernel_choice = mo.ui.dropdown(
        options=["Edge Detect (Horizontal)", "Edge Detect (Vertical)", "Sharpen", "Blur"],
        value="Edge Detect (Horizontal)",
        label="Kernel",
    )
    mo.hstack([conv_step, kernel_choice], justify="center", gap=1.5)
    return conv_step, kernel_choice


@app.cell
def _(conv_step, kernel_choice, mo):
    import numpy as _np
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    # 6x6 input "image"
    input_img = _np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
    ], dtype=float)

    kernels = {
        "Edge Detect (Horizontal)": _np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
        "Edge Detect (Vertical)": _np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
        "Sharpen": _np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
        "Blur": _np.ones((3, 3), dtype=float) / 9,
    }

    _kernel = kernels[kernel_choice.value]

    # Compute full convolution output (4x4 for 6x6 input with 3x3 kernel, no padding)
    out_size = 4
    output = _np.zeros((out_size, out_size))
    for _i in range(out_size):
        for _j in range(out_size):
            output[_i, _j] = _np.sum(input_img[_i:_i+3, _j:_j+3] * _kernel)

    # Current step position
    step = min(conv_step.value, out_size * out_size - 1)
    row, col = divmod(step, out_size)

    # Highlight the receptive field
    highlight = _np.full_like(input_img, _np.nan)
    highlight[row:row+3, col:col+3] = input_img[row:row+3, col:col+3]

    # Current computation
    patch = input_img[row:row+3, col:col+3]
    element_wise = patch * _kernel
    result = element_wise.sum()

    _fig = _make_subplots(
        rows=1, cols=3,
        subplot_titles=["Input (6x6)", "Kernel (3x3)", f"Output (4x4) — Step {step}"],
        horizontal_spacing=0.08,
    )

    # Input heatmap with receptive field highlight
    _fig.add_trace(
        _go.Heatmap(z=input_img[::-1], colorscale="Blues", showscale=False,
                   text=input_img[::-1].astype(int).astype(str), texttemplate="%{text}"),
        row=1, col=1,
    )
    # Draw rectangle for receptive field
    _fig.add_shape(
        type="rect", x0=col-0.5, x1=col+2.5, y0=5-row-2.5, y1=5-row+0.5,
        line=dict(color="#EF553B", width=3), row=1, col=1,
    )

    # Kernel
    _fig.add_trace(
        _go.Heatmap(z=_kernel[::-1], colorscale="RdBu", showscale=False, zmid=0,
                   text=_kernel[::-1].astype(str), texttemplate="%{text}"),
        row=1, col=2,
    )

    # Output (show computed values so far)
    output_display = _np.full((out_size, out_size), _np.nan)
    for s in range(step + 1):
        r, c = divmod(s, out_size)
        output_display[r, c] = output[r, c]

    _fig.add_trace(
        _go.Heatmap(z=output_display[::-1], colorscale="Viridis", showscale=False,
                   text=_np.where(_np.isnan(output_display[::-1]), "", output_display[::-1].astype(int).astype(str)),
                   texttemplate="%{text}"),
        row=1, col=3,
    )
    # Highlight current output position
    _fig.add_shape(
        type="rect", x0=col-0.5, x1=col+0.5, y0=out_size-1-row-0.5, y1=out_size-1-row+0.5,
        line=dict(color="#EF553B", width=3), row=1, col=3,
    )

    _fig.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=30, r=30, t=60, b=30),
    )

    # Computation detail
    patch_str = "\\begin{bmatrix}" + " \\\\ ".join([" & ".join(f"{v:.0f}" for v in r) for r in patch]) + "\\end{bmatrix}"
    kern_str = "\\begin{bmatrix}" + " \\\\ ".join([" & ".join(f"{v:.1f}" for v in r) for r in _kernel]) + "\\end{bmatrix}"

    mo.md(
        f"""
        **Position ({row}, {col}):** Element-wise multiply, then sum:

        $$\\text{{patch}} \\odot \\text{{kernel}} = {patch_str} \\odot {kern_str} = {result:.1f}$$
        """
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Filter Gallery: What Do Filters See?

    Different kernels detect different patterns. Apply preset filters to a real image
    to see the effect:
    """)
    return


@app.cell
def _(mo):
    filter_type = mo.ui.dropdown(
        options=["Original", "Horizontal Edges", "Vertical Edges", "Sharpen", "Blur", "Emboss"],
        value="Original",
        label="Filter Type",
    )
    filter_type
    return (filter_type,)


@app.cell
def _(filter_type):
    import numpy as _np
    from sklearn.datasets import load_digits as _load_digits
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    # Use an sklearn digit as our "image"
    _digits = _load_digits()
    img = _digits.images[0]  # 8x8 image of digit "0"

    _filters = {
        "Original": None,
        "Horizontal Edges": _np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
        "Vertical Edges": _np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
        "Sharpen": _np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
        "Blur": _np.ones((3, 3)) / 9,
        "Emboss": _np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
    }

    _kernel = _filters[filter_type.value]

    if _kernel is not None:
        # Apply convolution
        h, w = img.shape
        kh, kw = _kernel.shape
        out = _np.zeros((h - kh + 1, w - kw + 1))
        for _i in range(out.shape[0]):
            for _j in range(out.shape[1]):
                out[_i, _j] = _np.sum(img[_i:_i+kh, _j:_j+kw] * _kernel)
        filtered = out
    else:
        filtered = img

    _fig = _make_subplots(
        rows=1, cols=2,
        subplot_titles=["Original (8x8 digit)", f"After {filter_type.value}"],
    )

    _fig.add_trace(
        _go.Heatmap(z=img[::-1], colorscale="Gray", showscale=False),
        row=1, col=1,
    )
    _fig.add_trace(
        _go.Heatmap(z=filtered[::-1], colorscale="RdBu", showscale=False, zmid=0),
        row=1, col=2,
    )

    _fig.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. CNN Architecture Builder

    A CNN alternates between **convolutional layers** (detect features) and **pooling layers**
    (reduce spatial dimensions), then uses **fully connected layers** for classification.

    See how tensor shapes change through the network:
    """)
    return


@app.cell
def _(mo):
    arch_conv1 = mo.ui.slider(start=8, stop=64, step=8, value=32, label="Conv1 Filters")
    arch_conv2 = mo.ui.slider(start=16, stop=128, step=16, value=64, label="Conv2 Filters")
    arch_kernel = mo.ui.dropdown(
        options={"3x3": 3, "5x5": 5},
        value="3x3",
        label="Kernel Size",
    )
    mo.hstack([arch_conv1, arch_conv2, arch_kernel], justify="center", gap=1.5)
    return arch_conv1, arch_conv2, arch_kernel


@app.cell
def _(arch_conv1, arch_conv2, arch_kernel, mo):
    k = int(arch_kernel.value)
    c1 = arch_conv1.value
    c2 = arch_conv2.value

    # Compute shapes through the network (input: 28x28x1 like MNIST)
    # Conv1: 28x28x1 -> (28-k+1) x (28-k+1) x c1
    after_conv1 = 28 - k + 1
    # MaxPool: halve spatial dims
    after_pool1 = after_conv1 // 2
    # Conv2
    after_conv2 = after_pool1 - k + 1
    # MaxPool
    after_pool2 = after_conv2 // 2
    # Flatten
    flat_size = after_pool2 * after_pool2 * c2

    layers_info = [
        ("Input", f"28 x 28 x 1", 28*28),
        (f"Conv2d({k}x{k}, {c1})", f"{after_conv1} x {after_conv1} x {c1}", after_conv1*after_conv1*c1),
        ("ReLU", f"{after_conv1} x {after_conv1} x {c1}", after_conv1*after_conv1*c1),
        ("MaxPool2d(2)", f"{after_pool1} x {after_pool1} x {c1}", after_pool1*after_pool1*c1),
        (f"Conv2d({k}x{k}, {c2})", f"{after_conv2} x {after_conv2} x {c2}", after_conv2*after_conv2*c2),
        ("ReLU", f"{after_conv2} x {after_conv2} x {c2}", after_conv2*after_conv2*c2),
        ("MaxPool2d(2)", f"{after_pool2} x {after_pool2} x {c2}", after_pool2*after_pool2*c2),
        ("Flatten", f"{flat_size}", flat_size),
        ("Linear(128)", "128", 128),
        ("Linear(10)", "10 (classes)", 10),
    ]

    table_rows = "\n".join(
        f"| {name} | {shape} | {size:,} |"
        for name, shape, size in layers_info
    )

    # Parameter count
    conv1_params = (1 * c1 * k * k) + c1
    conv2_params = (c1 * c2 * k * k) + c2
    fc1_params = flat_size * 128 + 128
    fc2_params = 128 * 10 + 10
    total_params = conv1_params + conv2_params + fc1_params + fc2_params

    mo.md(
        f"""
        ### Shape Flow Through the CNN

        | Layer | Output Shape | Tensor Size |
        |-------|-------------|-------------|
        {table_rows}

        **Total parameters:** {total_params:,}
        - Conv layers: {conv1_params + conv2_params:,} ({(conv1_params + conv2_params)/total_params:.1%})
        - FC layers: {fc1_params + fc2_params:,} ({(fc1_params + fc2_params)/total_params:.1%})

        Notice: most parameters are in the fully connected layers, but the conv layers do the
        heavy lifting of feature extraction.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Train a CNN on Fashion-MNIST

    Let's build and train a CNN. We'll use sklearn's digits dataset (8x8 images)
    for speed, but the architecture is the same as you'd use for 28x28 MNIST.
    """)
    return


@app.cell
def _(mo):
    cnn_train_btn = mo.ui.run_button(label="Train CNN")
    cnn_train_btn
    return (cnn_train_btn,)


@app.cell
def _(cnn_train_btn, mo):
    import torch
    import torch.nn as _nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.model_selection import train_test_split
    import numpy as _np
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    cnn_train_btn.value

    # Load data
    _digits = _load_digits()
    X = _digits.images.astype(_np.float32)  # (1797, 8, 8)
    y = _digits.target.astype(_np.int64)

    # Add channel dimension: (N, 1, 8, 8)
    X = X[:, _np.newaxis, :, :]
    X = X / 16.0  # normalize

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # CNN for 8x8 images
    cnn_model = _nn.Sequential(
        _nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 8x8 -> 8x8
        _nn.ReLU(),
        _nn.MaxPool2d(2),  # 8x8 -> 4x4
        _nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 4x4 -> 4x4
        _nn.ReLU(),
        _nn.MaxPool2d(2),  # 4x4 -> 2x2
        _nn.Flatten(),
        _nn.Linear(32 * 2 * 2, 64),
        _nn.ReLU(),
        _nn.Linear(64, 10),
    )

    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    criterion = _nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(15):
        cnn_model.train()
        epoch_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            pred = cnn_model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            correct += (pred.argmax(1) == yb).sum().item()
            total += len(xb)
        train_losses.append(epoch_loss / total)
        train_accs.append(correct / total)

        cnn_model.eval()
        epoch_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = cnn_model(xb)
                loss = criterion(pred, yb)
                epoch_loss += loss.item() * len(xb)
                correct += (pred.argmax(1) == yb).sum().item()
                total += len(xb)
        val_losses.append(epoch_loss / total)
        val_accs.append(correct / total)

    # Visualize first-layer filters
    _filters = cnn_model[0].weight.detach().numpy()  # (16, 1, 3, 3)

    _fig = _make_subplots(
        rows=2, cols=2,
        subplot_titles=["Training Curves", "Learned Filters (Conv1)", "Sample Predictions", ""],
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    epochs_range = list(range(1, len(train_losses) + 1))
    _fig.add_trace(_go.Scatter(x=epochs_range, y=train_accs, name="Train Acc", line=dict(color="#636EFA")), row=1, col=1)
    _fig.add_trace(_go.Scatter(x=epochs_range, y=val_accs, name="Val Acc", line=dict(color="#EF553B")), row=1, col=1)

    # Show 16 filters as a 4x4 grid
    filter_grid = _np.zeros((4 * 3, 4 * 3))
    for _i in range(4):
        for _j in range(4):
            idx = _i * 4 + _j
            filter_grid[_i*3:(_i+1)*3, _j*3:(_j+1)*3] = _filters[idx, 0]

    _fig.add_trace(
        _go.Heatmap(z=filter_grid[::-1], colorscale="RdBu", showscale=False, zmid=0),
        row=1, col=2,
    )

    # Sample predictions
    cnn_model.eval()
    with torch.no_grad():
        sample_x = torch.FloatTensor(X_val[:10])
        sample_pred = cnn_model(sample_x).argmax(1).numpy()
        sample_true = y_val[:10]

    pred_text = " | ".join(
        f"{'✓' if p == t else '✗'} {p}" for p, t in zip(sample_pred, sample_true)
    )

    _fig.update_layout(
        template="plotly_dark", height=500,
        margin=dict(l=30, r=30, t=60, b=30),
        title=f"CNN Results | Val Accuracy: {val_accs[-1]:.1%}",
    )

    mo.md(
        f"""
        **Predictions on 10 validation samples:** {pred_text}

        *True labels: {list(sample_true)}*

        The 4x4 grid shows the 16 learned 3x3 filters from the first conv layer.
        Notice they've learned different edge orientations and patterns —
        similar to hand-crafted edge detection filters, but learned from data.
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
    | **Convolution** | Sliding a kernel to detect local patterns — element-wise multiply + sum |
    | **Filters/Kernels** | Learned feature detectors — edges, textures, shapes |
    | **Pooling** | Spatial downsampling — reduces size, adds translation invariance |
    | **CNN architecture** | Conv → ReLU → Pool → ... → Flatten → FC → Output |
    | **Learned filters** | CNNs discover their own features from data |

    ### Key Takeaway
    CNNs exploit the spatial structure of images by using local, shared filters.
    Instead of connecting every pixel to every neuron (MLP), convolutions focus on
    local neighborhoods and reuse the same weights across the image. This is why
    CNNs need far fewer parameters than MLPs for image tasks.

    **Next up:** [Notebook 04 — Sequence Models](04_sequence_models.py) —
    how do we handle sequential data like text and time series?
    """)
    return


if __name__ == "__main__":
    app.run()
