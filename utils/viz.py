"""Shared visualization helpers for deep learning notebooks."""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Consistent plotly theme
PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    "primary": "#636EFA",
    "secondary": "#EF553B",
    "tertiary": "#00CC96",
    "quaternary": "#AB63FA",
    "quinary": "#FFA15A",
    "train": "#636EFA",
    "val": "#EF553B",
    "positive": "#00CC96",
    "negative": "#EF553B",
}


def plotly_layout(**kwargs):
    """Return a standard plotly layout with dark theme."""
    defaults = dict(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=50, r=30, t=50, b=50),
        font=dict(size=12),
    )
    defaults.update(kwargs)
    return go.Layout(**defaults)


def plot_network(layers: list[int], title: str = "Network Architecture") -> go.Figure:
    """Draw a neural network architecture diagram as a plotly figure.

    Args:
        layers: List of neuron counts per layer, e.g. [2, 4, 3, 1]
        title: Plot title
    """
    fig = go.Figure()
    max_neurons = max(layers)
    x_spacing = 1.5
    node_x, node_y, node_text = [], [], []

    for layer_idx, n_neurons in enumerate(layers):
        x = layer_idx * x_spacing
        y_positions = np.linspace(-(n_neurons - 1) / 2, (n_neurons - 1) / 2, n_neurons)

        for neuron_idx, y in enumerate(y_positions):
            node_x.append(x)
            node_y.append(y)
            if layer_idx == 0:
                node_text.append(f"Input {neuron_idx + 1}")
            elif layer_idx == len(layers) - 1:
                node_text.append(f"Output {neuron_idx + 1}")
            else:
                node_text.append(f"L{layer_idx} N{neuron_idx + 1}")

        # Draw connections to previous layer
        if layer_idx > 0:
            prev_n = layers[layer_idx - 1]
            prev_x = (layer_idx - 1) * x_spacing
            prev_y_positions = np.linspace(
                -(prev_n - 1) / 2, (prev_n - 1) / 2, prev_n
            )
            for py in prev_y_positions:
                for cy in y_positions:
                    fig.add_trace(
                        go.Scatter(
                            x=[prev_x, x],
                            y=[py, cy],
                            mode="lines",
                            line=dict(color="rgba(150,150,150,0.3)", width=1),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

    # Draw nodes on top
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=25, color=COLORS["primary"], line=dict(width=2, color="white")),
            text=node_text,
            textposition="top center",
            textfont=dict(size=8),
            showlegend=False,
        )
    )

    # Layer labels
    for i, n in enumerate(layers):
        label = "Input" if i == 0 else ("Output" if i == len(layers) - 1 else f"Hidden {i}")
        fig.add_annotation(
            x=i * x_spacing,
            y=max_neurons / 2 + 0.8,
            text=f"{label}<br>({n} neurons)",
            showarrow=False,
            font=dict(size=10),
        )

    fig.update_layout(
        plotly_layout(
            title=title,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x"),
            height=400,
        )
    )
    return fig


def plot_attention_heatmap(
    weights: np.ndarray, tokens: list[str], title: str = "Attention Weights"
) -> go.Figure:
    """Interactive attention heatmap.

    Args:
        weights: 2D array of attention weights (query x key)
        tokens: List of token strings
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=weights,
            x=tokens,
            y=tokens,
            colorscale="Viridis",
            text=np.round(weights, 3),
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        plotly_layout(
            title=title,
            xaxis_title="Key",
            yaxis_title="Query",
            yaxis=dict(autorange="reversed"),
            height=450,
            width=500,
        )
    )
    return fig


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    train_accs: list[float] | None = None,
    val_accs: list[float] | None = None,
    title: str = "Training Progress",
) -> go.Figure:
    """Plot training and validation loss/accuracy curves.

    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch (optional)
        train_accs: Training accuracy per epoch (optional)
        val_accs: Validation accuracy per epoch (optional)
    """
    from plotly.subplots import make_subplots

    has_acc = train_accs is not None
    fig = make_subplots(
        rows=1,
        cols=2 if has_acc else 1,
        subplot_titles=["Loss", "Accuracy"] if has_acc else ["Loss"],
    )
    epochs = list(range(1, len(train_losses) + 1))

    fig.add_trace(
        go.Scatter(x=epochs, y=train_losses, name="Train Loss", line=dict(color=COLORS["train"])),
        row=1, col=1,
    )
    if val_losses:
        fig.add_trace(
            go.Scatter(x=epochs, y=val_losses, name="Val Loss", line=dict(color=COLORS["val"])),
            row=1, col=1,
        )

    if has_acc:
        fig.add_trace(
            go.Scatter(x=epochs, y=train_accs, name="Train Acc", line=dict(color=COLORS["train"])),
            row=1, col=2,
        )
        if val_accs:
            fig.add_trace(
                go.Scatter(x=epochs, y=val_accs, name="Val Acc", line=dict(color=COLORS["val"])),
                row=1, col=2,
            )

    fig.update_layout(plotly_layout(title=title, height=350))
    fig.update_xaxes(title_text="Epoch")
    return fig


def plot_gradient_flow(named_parameters) -> go.Figure:
    """Plot gradient magnitudes per layer to diagnose vanishing/exploding gradients.

    Args:
        named_parameters: Iterator of (name, param) from model.named_parameters()
    """
    layers, avg_grads, max_grads = [], [], []
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            avg_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=layers, y=max_grads, name="Max Gradient", marker_color=COLORS["secondary"], opacity=0.5)
    )
    fig.add_trace(
        go.Bar(x=layers, y=avg_grads, name="Avg Gradient", marker_color=COLORS["primary"])
    )
    fig.update_layout(
        plotly_layout(
            title="Gradient Flow",
            xaxis_title="Layer",
            yaxis_title="Gradient Magnitude",
            yaxis_type="log",
            barmode="overlay",
            height=350,
        )
    )
    return fig
