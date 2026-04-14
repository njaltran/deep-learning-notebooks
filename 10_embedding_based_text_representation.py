import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import math
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    return go, math, np


@app.cell
def _(mo):
    mo.md(r"""
    # Text Representation — Part 2: Embedding-Based Approaches

    **NLP Lab · Prof. Dr. Diana Hristova**

    ---

    ## Learning Objectives

    By the end of this notebook you will:

    1. Know the main idea behind using **embeddings** as document representation.
    2. Understand the **Word2Vec** and **ELMo** approaches for document representation.
    3. Be able to compute **embedding similarity**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1  Motivation: Why Embeddings?

    ### Recap — Bag-of-Words

    The Bag-of-Words (BoW) approach represents each document as a vector where
    **each feature = one word in the vocabulary**, and feature values are word
    importance scores (e.g. raw counts or TF-IDF weights).
    """)
    return


@app.cell
def _(mo):
    # Interactive BoW demo — seeded with the lecture's movie example
    doc1_input = mo.ui.text(value="movie was great", label="Doc 1")
    doc2_input = mo.ui.text(value="movie was great was amazing", label="Doc 2")
    doc3_input = mo.ui.text(value="movie was bad", label="Doc 3")

    mo.md(f"""
    ### Try it: Build a BoW Table

    The defaults match the lecture's movie example. Edit them and watch the BoW matrix update live.

    {mo.hstack([doc1_input, doc2_input, doc3_input])}
    """)
    return doc1_input, doc2_input, doc3_input


@app.cell
def _(doc1_input, doc2_input, doc3_input, mo):
    def build_bow(docs):
        vocab = sorted(set(w for d in docs for w in d.lower().split() if w))
        rows = []
        for i, doc in enumerate(docs, 1):
            words = doc.lower().split()
            row = {"Doc": f'Doc{i}: "{doc}"'}
            for v in vocab:
                row[v] = words.count(v)
            rows.append(row)
        return vocab, rows

    raw_docs = [doc1_input.value, doc2_input.value]
    if doc3_input.value.strip():
        raw_docs.append(doc3_input.value)

    vocab, bow_rows = build_bow(raw_docs)

    header = "| Doc | " + " | ".join(vocab) + " |"
    sep = "|---|" + "|".join(["---"] * len(vocab)) + "|"
    rows_md = []
    for r in bow_rows:
        cells = [str(r.get(v, 0)) for v in vocab]
        rows_md.append("| " + r["Doc"] + " | " + " | ".join(cells) + " |")

    table_md = "\n".join([header, sep] + rows_md)

    mo.md(f"""
    #### BoW Representation

    {table_md}

    > **Lecture example:** BoW only records how often each vocabulary item appears in each document.
    >
    > **What it captures well:** Doc1 and Doc2 look similar because they share `movie`, `was`, and `great`.
    >
    > **What it still misses:** `great` and `amazing` are different columns, and word order is ignored.
    > `"movie was great"` and `"great movie was"` would get the same vector.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2  What Are Embeddings?

    **Embeddings (distributional representations)** map each word into a *continuous*
    vector space so that **semantically similar words end up geometrically close**.

    Key ideas:
    - Words that appear in **similar contexts** get similar vectors.
    - Vectors can have **negative values** — this is essential for encoding
      opposition (e.g. *horrible* vs *wonderful*).
    - The dimensionality is a **hyperparameter** (typically 50–300 for Word2Vec,
      768+ for Transformer models).

    > *"You shall know a word by the company it keeps."* — J.R. Firth (1957)
    """)
    return


@app.cell
def _(go):
    # 2-D vector space illustration (from lecture slide 10)
    words_2d = {
        "wonderful": (5, 1),
        "adorable":  (5, 4),
        "horrible":  (-5, 3),
        "terrible":  (-4, 1),
    }

    fig_space = go.Figure()
    colors = {"wonderful": "#2196F3", "adorable": "#4CAF50",
              "horrible": "#F44336", "terrible": "#FF9800"}

    for word, (x, y) in words_2d.items():
        fig_space.add_trace(go.Scatter(
            x=[0, x], y=[0, y],
            mode="lines+markers+text",
            line=dict(color=colors[word], width=2),
            marker=dict(size=[4, 10], color=colors[word]),
            text=["", word],
            textposition="top center",
            textfont=dict(size=13, color=colors[word]),
            name=word,
            showlegend=True,
        ))

    fig_space.update_layout(
        title="2-D Embedding Space: Similar Words Cluster Together",
        xaxis=dict(zeroline=True, range=[-7, 8], title="Dimension 1"),
        yaxis=dict(zeroline=True, range=[-1, 6], title="Dimension 2"),
        height=380,
        margin=dict(t=50, b=40),
    )
    fig_space
    return (words_2d,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3  Word2Vec

    **Word2Vec** (Mikolov et al., Google 2013) is the classic embedding model and can be understood as a shallow **two-layer neural network**.

    ### Core Idea
    *Words that appear in similar contexts must have similar embedding representations.*

    ### Two Architectures

    | Architecture | Input → Output |
    |---|---|
    | **CBOW** (Continuous Bag-of-Words) | surrounding context words → predict target word |
    | **Skip-gram** | target word → predict surrounding context words |

    ### Why people call it a 2-layer neural network

    For a skip-gram example with input word `meal`:

    1. The **input / embedding layer** multiplies the one-hot vector by an embedding matrix `W_1` and selects the hidden representation `r_{meal}`.
    2. The **output layer** multiplies `r_{meal}` by a second weight matrix `W_2` to produce one score per vocabulary word.
    3. A **softmax** converts those scores into prediction probabilities for the surrounding context words.

    **Skip-gram training** (window = 2 around the target word *w*):

    1. Start with random embedding $r_w$ and context representation $r_{c,w}$ for each word.
    2. For every other word $u$, calculate the probability it appears near $w$:
       $$p(u \mid w) \propto \exp\bigl(\text{similarity}(r_{c,u},\, r_w)\bigr)$$
    3. Adjust both representations so predicted probabilities match the corpus.

    After training, *similar words end up with similar $r_w$ vectors*.
    """)
    return


@app.cell
def _(go, mo):
    # Full Word2Vec skip-gram network flow for one training example
    arch_vocab = ["fantastic", "service", "meal", "great"]
    arch_one_hot = [0, 0, 1, 0]  # input word = "meal"
    arch_hidden = [0.40, -0.10, 0.80]
    arch_probs = [0.20, 0.44, 0.15, 0.22]

    fig_word2vec_arch = go.Figure()


    def w2v_add_box(x0, y0, x1, y1, text, fill, line="#37474F", font_size=11):
        fig_word2vec_arch.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=line, width=2),
            fillcolor=fill,
            layer="below",
        )
        fig_word2vec_arch.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=text,
            showarrow=False,
            align="center",
            font=dict(size=font_size),
        )


    def w2v_add_arrow(x0, y0, x1, y1, color="#546E7A"):
        fig_word2vec_arch.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.8, arrowcolor=color,
        )


    arch_y_positions = [3.4, 2.5, 1.6, 0.7]

    for arch_word, arch_value, arch_y in zip(arch_vocab, arch_one_hot, arch_y_positions):
        arch_fill = "#BBDEFB" if arch_value == 1 else "#ECEFF1"
        w2v_add_box(-0.1, arch_y - 0.28, 1.1, arch_y + 0.28, f"{arch_word}<br>{arch_value}", arch_fill)

    w2v_add_box(2.0, 0.35, 3.6, 3.75, "Embedding matrix<br><b>W₁</b><br>shape = 4 × 3", "#FFF3E0")
    for arch_idx, (arch_value, arch_y) in enumerate(zip(arch_hidden, [2.9, 2.2, 1.5]), 1):
        fig_word2vec_arch.add_trace(go.Scatter(
            x=[5.0], y=[arch_y],
            mode="markers+text",
            marker=dict(size=34, color="#FFE082", line=dict(color="#FB8C00", width=2)),
            text=[f"h{arch_idx}<br>{arch_value:.2f}"],
            textposition="middle center",
            textfont=dict(size=10),
            showlegend=False,
        ))

    w2v_add_box(6.3, 0.35, 7.9, 3.75, "Context matrix<br><b>W₂</b><br>shape = 3 × 4", "#E8F5E9")

    arch_linear_scores = ["score(fantastic)", "score(service)", "score(meal)", "score(great)"]
    for arch_label, arch_y in zip(arch_linear_scores, arch_y_positions):
        w2v_add_box(8.6, arch_y - 0.28, 10.0, arch_y + 0.28, arch_label, "#F3E5F5", font_size=10)

    for arch_word, arch_prob, arch_y in zip(arch_vocab, arch_probs, arch_y_positions):
        w2v_add_box(11.0, arch_y - 0.28, 12.4, arch_y + 0.28, f"{arch_word}<br>{arch_prob:.2f}", "#E1F5FE")

    w2v_add_arrow(1.15, 2.0, 1.95, 2.0)
    w2v_add_arrow(3.65, 2.0, 4.55, 2.2)
    w2v_add_arrow(5.45, 2.2, 6.25, 2.2)
    w2v_add_arrow(7.95, 2.2, 8.55, 2.2)
    w2v_add_arrow(10.05, 2.2, 10.95, 2.2)

    fig_word2vec_arch.add_annotation(
        x=0.5, y=4.25, text="One-hot input<br>dictionary size = 4",
        showarrow=False, font=dict(size=12, color="#1565C0"),
    )
    fig_word2vec_arch.add_annotation(
        x=5.0, y=4.25, text='Hidden layer (size 3)<br><b>= r<sub>meal</sub></b>',
        showarrow=False, font=dict(size=12, color="#E65100"),
    )
    fig_word2vec_arch.add_annotation(
        x=9.3, y=4.25, text="Linear output scores",
        showarrow=False, font=dict(size=12, color="#6A1B9A"),
    )
    fig_word2vec_arch.add_annotation(
        x=11.7, y=4.25, text="Softmax probabilities",
        showarrow=False, font=dict(size=12, color="#0277BD"),
    )
    fig_word2vec_arch.add_annotation(
        x=9.0, y=0.08,
        text="score(u) = r<sub>c,u</sub> · r<sub>meal</sub><br>e.g. similarity(r<sub>c,fantastic</sub>, r<sub>meal</sub>)",
        showarrow=False,
        font=dict(size=11),
        align="center",
    )
    fig_word2vec_arch.add_annotation(
        x=2.8, y=4.0, text="select / lookup", showarrow=False, font=dict(size=10, color="#546E7A")
    )
    fig_word2vec_arch.add_annotation(
        x=10.55, y=2.85, text="softmax", showarrow=False, font=dict(size=10, color="#546E7A")
    )

    fig_word2vec_arch.update_layout(
        title='Word2Vec Skip-gram as a Two-Layer Neural Network (input word: "meal")',
        xaxis=dict(range=[-0.6, 12.8], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.4, 4.7], showgrid=False, zeroline=False, showticklabels=False),
        height=520,
        margin=dict(t=70, b=30, l=20, r=20),
        plot_bgcolor="white",
    )

    mo.vstack([
        fig_word2vec_arch,
        mo.md(r"""
        The hidden layer is the learned embedding vector `r_meal`.
        `W_2` maps that vector to vocabulary scores, and the softmax layer turns them into context-word probabilities.
        """),
    ])
    return


@app.cell
def _(go):
    # Skip-gram prediction illustration (lecture slide 14)
    mini_vocab = ["fantastic", "service", "meal", "great"]
    # Predicted output layer (softmax) for input "meal"
    predicted = [0.20, 0.44, 0.15, 0.22]  # from slide
    # Ground truth: context of "meal" with window=2 in "service great meal" + "service fantastic meal"
    ground_truth = [0.25, 0.50, 0.00, 0.25]

    fig_skipgram = go.Figure(data=[
        go.Bar(name="Ground Truth (corpus)", x=mini_vocab, y=ground_truth,
               marker_color="#4CAF50", opacity=0.8),
        go.Bar(name="Model Prediction (softmax)", x=mini_vocab, y=predicted,
               marker_color="#2196F3", opacity=0.8),
    ])
    fig_skipgram.update_layout(
        barmode="group",
        title='Skip-gram: Context of "meal" (window=2) — Ground Truth vs Prediction',
        yaxis=dict(title="Probability", tickformat=".0%"),
        height=340,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    fig_skipgram
    return


@app.cell
def _(mo):
    mo.md(r"""
    > **Key take-away:** The model adjusts the embedding weights until the predicted
    > context distribution matches the actual co-occurrence statistics in the corpus.
    > The hidden-layer weights **become** the word embeddings.
    >
    > The grouped bar chart above is the **final softmax output** of that two-layer network for input word `meal`.

    ### Remarkable property — vector arithmetic
    Because meaning is encoded geometrically, you can do algebra:

    $$\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$$
    """)
    return


@app.cell
def _(go, np):
    # Analogy demo in 2-D (approximate, illustrative)
    analogy_words = {
        "Man":   np.array([1.5, 3.5]),
        "Woman": np.array([1.5, 1.5]),
        "King":  np.array([4.5, 3.5]),
        "Queen": np.array([4.5, 1.5]),
    }
    predicted_queen = analogy_words["King"] - analogy_words["Man"] + analogy_words["Woman"]

    fig_analogy = go.Figure()
    colors_a = {"Man": "#607D8B", "Woman": "#E91E63", "King": "#9C27B0", "Queen": "#FF9800"}
    for analogy_word, analogy_vec in analogy_words.items():
        fig_analogy.add_trace(go.Scatter(
            x=[analogy_vec[0]], y=[analogy_vec[1]],
            mode="markers+text",
            marker=dict(size=16, color=colors_a[analogy_word]),
            text=[analogy_word], textposition="top center",
            textfont=dict(size=12),
            name=analogy_word,
        ))

    # Arrow: King - Man + Woman = Queen
    for analogy_start, analogy_end in [
        (analogy_words["Man"], analogy_words["King"]),
        (analogy_words["Woman"], predicted_queen),
    ]:
        fig_analogy.add_annotation(
            ax=analogy_start[0], ay=analogy_start[1], x=analogy_end[0], y=analogy_end[1],
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowwidth=2, arrowcolor="#555",
        )

    fig_analogy.update_layout(
        title="Word2Vec Analogy: King − Man + Woman ≈ Queen",
        xaxis=dict(range=[0, 6], title="Dim 1"),
        yaxis=dict(range=[0, 5], title="Dim 2"),
        height=360,
        margin=dict(t=50),
        showlegend=True,
    )
    fig_analogy
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4  Embedding Similarity

    ### 4.1 Cosine Similarity

    The standard metric for comparing embedding vectors is **cosine similarity** —
    the cosine of the angle between them:

    $$\text{CosSim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2}\;\sqrt{\sum_i b_i^2}}$$

    | Value | Interpretation |
    |---|---|
    | **+1** | identical direction (cos 0°) |
    | **0** | orthogonal / unrelated (cos 90°) |
    | **−1** | opposite direction (cos 180°) |
    """)
    return


@app.cell
def _(mo):
    # Interactive cosine similarity calculator
    ax = mo.ui.slider(-8, 8, value=5, step=0.5, label="Word A — x")
    ay = mo.ui.slider(-8, 8, value=1, step=0.5, label="Word A — y")
    bx = mo.ui.slider(-8, 8, value=5, step=0.5, label="Word B — x")
    by = mo.ui.slider(-8, 8, value=4, step=0.5, label="Word B — y")
    word_a_name = mo.ui.text(value="wonderful", label="Name A")
    word_b_name = mo.ui.text(value="adorable",  label="Name B")

    mo.md(f"""
    ### Interactive Cosine Similarity Calculator

    Drag the sliders to set 2-D word vectors, or use the lecture examples:
    - *wonderful* = (5, 1), *adorable* = (5, 4) → CosSim ≈ 0.89
    - *wonderful* = (5, 1), *horrible* = (−5, 3) → CosSim ≈ −0.74

    {mo.hstack([word_a_name, word_b_name])}
    {mo.hstack([ax, ay])}
    {mo.hstack([bx, by])}
    """)
    return ax, ay, bx, by, word_a_name, word_b_name


@app.cell
def _(ax, ay, bx, by, go, math, mo, word_a_name, word_b_name):
    def cosine_sim(a1, a2, b1, b2):
        dot = a1 * b1 + a2 * b2
        norm_a = math.sqrt(a1**2 + a2**2)
        norm_b = math.sqrt(b1**2 + b2**2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    cs = cosine_sim(ax.value, ay.value, bx.value, by.value)
    angle_deg = math.degrees(math.acos(max(-1, min(1, cs))))

    fig_cos = go.Figure()
    for vec_name, vx, vy, vec_color in [
        (word_a_name.value, ax.value, ay.value, "#2196F3"),
        (word_b_name.value, bx.value, by.value, "#F44336"),
    ]:
        fig_cos.add_trace(go.Scatter(
            x=[0, vx], y=[0, vy],
            mode="lines+markers+text",
            line=dict(color=vec_color, width=3),
            marker=dict(size=[4, 12], color=vec_color),
            text=["", vec_name],
            textposition="top center",
            textfont=dict(size=13, color=vec_color),
            name=vec_name,
        ))

    lim = max(abs(ax.value), abs(ay.value), abs(bx.value), abs(by.value), 2) + 1
    fig_cos.update_layout(
        title=f"CosSim({word_a_name.value}, {word_b_name.value}) = {cs:.4f}  (angle = {angle_deg:.1f}°)",
        xaxis=dict(zeroline=True, range=[-lim, lim]),
        yaxis=dict(zeroline=True, range=[-lim, lim], scaleanchor="x"),
        height=400,
        margin=dict(t=60, b=40),
    )

    if cs > 0.7:
        interpretation = "Very similar (small angle)"
        badge_color = "green"
    elif cs > 0.3:
        interpretation = "Somewhat similar"
        badge_color = "blue"
    elif cs > -0.3:
        interpretation = "Unrelated / orthogonal"
        badge_color = "gray"
    elif cs > -0.7:
        interpretation = "Somewhat opposite"
        badge_color = "orange"
    else:
        interpretation = "Very dissimilar / opposite"
        badge_color = "red"

    mo.vstack([
        fig_cos,
        mo.md(f"""
        **Result:** CosSim = `{cs:.4f}`  |  Angle = `{angle_deg:.1f}°`  |  *{interpretation}*

        **Calculation:**
        $$\\text{{CosSim}} = \\frac{{({ax.value})({bx.value}) + ({ay.value})({by.value})}}
        {{\\sqrt{{{ax.value}^2 + {ay.value}^2}}\\;\\sqrt{{{bx.value}^2 + {by.value}^2}}}} = {cs:.4f}$$
        """),
    ])
    cosine_sim
    return (cosine_sim,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.2 Other Similarity / Distance Measures

    | Measure | Formula | Notes |
    |---|---|---|
    | **Dot product** | $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$ | Cosine without normalisation — biased by vector magnitude |
    | **Euclidean distance** | $\|\mathbf{a} - \mathbf{b}\| = \sqrt{\sum_i (a_i - b_i)^2}$ | Straight-line distance — also sensitive to magnitude |
    | **Cosine similarity** | $\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$ | Magnitude-invariant — **preferred for NLP** |

    > **Why cosine for NLP?**
    > A 500-word document and a 5-word document may cover the same topic.
    > Cosine similarity ignores length; dot product and Euclidean distance do not.
    """)
    return


@app.cell
def _(cosine_sim, go, math, words_2d):
    # Comparison of all three measures for the lecture's example words
    example_pairs = [
        ("wonderful", "adorable"),
        ("wonderful", "horrible"),
        ("adorable",  "horrible"),
        ("wonderful", "terrible"),
    ]
    measure_names = ["Cosine Similarity", "Dot Product", "Euclidean Distance"]
    results = {measure_name: [] for measure_name in measure_names}
    pair_labels = []

    for w1, w2 in example_pairs:
        a = words_2d[w1]
        b = words_2d[w2]
        pair_labels.append(f"{w1}\\nvs\\n{w2}")
        results["Cosine Similarity"].append(cosine_sim(a[0], a[1], b[0], b[1]))
        results["Dot Product"].append(a[0] * b[0] + a[1] * b[1])
        results["Euclidean Distance"].append(
            math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        )

    fig_measures = go.Figure()
    colors_m = ["#2196F3", "#4CAF50", "#F44336"]
    for measure_idx, (measure_name, measure_values) in enumerate(results.items()):
        fig_measures.add_trace(go.Bar(
            name=measure_name, x=pair_labels, y=measure_values,
            marker_color=colors_m[measure_idx], opacity=0.8,
        ))

    fig_measures.update_layout(
        barmode="group",
        title="Similarity Measures Compared (lecture example vectors)",
        yaxis_title="Value",
        height=360,
        margin=dict(t=50, b=60),
        legend=dict(orientation="h", y=1.1),
    )
    fig_measures
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5  Word2Vec: Pros & Cons

    | | |
    |---|---|
    | **Advantages** | Captures word meaning via context; supports vector arithmetic; interpretable geometry |
    | **Disadvantage** | **Static** — the same vector regardless of context |

    ### The Polysemy Problem

    The word *bank* has very different meanings depending on context:

    - "I need to go to the **bank** office" → financial institution
    - "We sat on the river **bank**" → geographic feature
    - "The **blood bank** was overwhelmed" → storage facility

    Word2Vec assigns *one single vector* to "bank" — it ends up as a confused
    average of all usages.

    → Solution: **Contextual Embeddings** (ELMo, BERT, GPT-2…)
    """)
    return


@app.cell
def _(go, np):
    # Illustrate static vs contextual embeddings for "bank"
    np.random.seed(42)

    bank_finance = np.array([3.0, 4.0])
    bank_river   = np.array([-3.5, 2.0])
    bank_blood   = np.array([1.0, -3.0])
    bank_static  = (bank_finance + bank_river + bank_blood) / 3

    fig_polysemy = go.Figure()

    for sense_label, sense_point, sense_color in [
        ("bank (financial)", bank_finance, "#2196F3"),
        ("bank (river)",     bank_river,   "#4CAF50"),
        ("bank (blood)",     bank_blood,   "#F44336"),
    ]:
        fig_polysemy.add_trace(go.Scatter(
            x=[sense_point[0]], y=[sense_point[1]],
            mode="markers+text",
            marker=dict(size=18, color=sense_color, symbol="circle"),
            text=[sense_label], textposition="top center",
            textfont=dict(size=11, color=sense_color),
            name=sense_label,
        ))

    fig_polysemy.add_trace(go.Scatter(
        x=[bank_static[0]], y=[bank_static[1]],
        mode="markers+text",
        marker=dict(size=20, color="#FF9800", symbol="star"),
        text=["bank (Word2Vec — static average)"],
        textposition="bottom right",
        textfont=dict(size=11, color="#FF9800"),
        name="bank (Word2Vec static)",
    ))

    for sense_point in [bank_finance, bank_river, bank_blood]:
        fig_polysemy.add_trace(go.Scatter(
            x=[bank_static[0], sense_point[0]],
            y=[bank_static[1], sense_point[1]],
            mode="lines",
            line=dict(dash="dash", color="#FF9800", width=1),
            showlegend=False,
        ))

    fig_polysemy.update_layout(
        title='Polysemy Problem: Word2Vec assigns one vector to "bank"',
        xaxis=dict(zeroline=True, range=[-6, 6]),
        yaxis=dict(zeroline=True, range=[-5, 6]),
        height=420,
        margin=dict(t=50),
    )
    fig_polysemy
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6  Contextual Embeddings: ELMo

    **ELMo** (Embeddings from Language Models) was developed by AI2 (Peters et al., 2018).

    ### Key Differences from Word2Vec

    | | Word2Vec | ELMo |
    |---|---|---|
    | Embedding type | Static (one per word) | **Dynamic** (recomputed per sentence) |
    | Context window | Small (2–5 words) | **Full sentence** |
    | Handles polysemy? | No | **Yes** |
    | Tokenisation | Word-level | **Character-level** → handles OOV words |
    | Architecture | Shallow 2-layer neural net | **2-layer bi-directional LSTM language model** |

    ### Architecture

    ELMo is trained as a **bi-directional language model**.
    The forward LM predicts the next word given all preceding words, and the backward LM predicts the previous word given all following words.

    Each LSTM layer produces a **hidden state** $h_{w,k}$ per position $w$ and layer $k$.
    The final ELMo embedding combines all layers:

    $$\text{ELMo}_w = \gamma \sum_{k=0}^{K} s_k \cdot h_{w,k}$$

    where $s_k$ are learned task-specific weights and $\gamma$ is a scaling factor.

    In practice, you usually start from a **pre-trained ELMo model** and then fine-tune the task-specific layer weights and downstream prediction head for your problem.

    During language-model pre-training, the top hidden states feed a **linear + softmax layer** that produces prediction probabilities:

    $$p(w_t \mid \text{context}) = \operatorname{softmax}(W h_t + b)$$
    """)
    return


@app.cell
def _(go):
    # ELMo bi-directional LSTM diagram with language-model prediction head
    tokens = ["the", "cat", "is", "happy"]
    n = len(tokens)

    fig_elmo = go.Figure()


    def elmo_add_box(fig, x, y, label, color, size=20, text_color="white"):
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=size, color=color, symbol="square"),
            text=[label], textposition="middle center",
            textfont=dict(size=9, color=text_color),
            showlegend=False,
        ))


    def elmo_add_arrow(fig, x0, y0, x1, y1, color="#555"):
        fig.add_annotation(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowwidth=1.5, arrowcolor=color, arrowsize=0.8,
        )


    xs = [1, 2, 3, 4]

    for xi, tok in zip(xs, tokens):
        elmo_add_box(fig_elmo, xi, 0.0, tok, "#607D8B", size=24)

    for xi in xs:
        elmo_add_box(fig_elmo, xi, 1.2, "LSTM₁→", "#1976D2", size=22)
        elmo_add_box(fig_elmo, xi, 0.7, "LSTM₁←", "#C62828", size=22)
    for forward_layer1_idx in range(n - 1):
        elmo_add_arrow(fig_elmo, xs[forward_layer1_idx], 1.2, xs[forward_layer1_idx + 1], 1.2, "#1976D2")
    for backward_layer1_idx in range(n - 1, 0, -1):
        elmo_add_arrow(fig_elmo, xs[backward_layer1_idx], 0.7, xs[backward_layer1_idx - 1], 0.7, "#C62828")

    for xi in xs:
        elmo_add_box(fig_elmo, xi, 2.7, "LSTM₂→", "#0D47A1", size=22)
        elmo_add_box(fig_elmo, xi, 2.2, "LSTM₂←", "#B71C1C", size=22)
    for forward_layer2_idx in range(n - 1):
        elmo_add_arrow(fig_elmo, xs[forward_layer2_idx], 2.7, xs[forward_layer2_idx + 1], 2.7, "#0D47A1")
    for backward_layer2_idx in range(n - 1, 0, -1):
        elmo_add_arrow(fig_elmo, xs[backward_layer2_idx], 2.2, xs[backward_layer2_idx - 1], 2.2, "#B71C1C")

    for xi in xs:
        elmo_add_box(fig_elmo, xi, 4.0, "Linear<br>+ softmax", "#F9A825", size=24, text_color="black")

    for xi in xs:
        elmo_add_arrow(fig_elmo, xi, 0.2, xi, 0.55, "#555")
        elmo_add_arrow(fig_elmo, xi, 0.9, xi, 1.05, "#555")
        elmo_add_arrow(fig_elmo, xi, 1.35, xi, 2.05, "#555")
        elmo_add_arrow(fig_elmo, xi, 2.35, xi, 2.55, "#555")
        elmo_add_arrow(fig_elmo, xi - 0.08, 2.85, xi, 3.75, "#555")
        elmo_add_arrow(fig_elmo, xi + 0.08, 2.35, xi, 3.75, "#555")

    for elmo_y_pos, elmo_label, elmo_col in [
        (0.0, "Input tokens / character CNN embeddings", "#607D8B"),
        (1.2, "Forward LSTM layer 1", "#1976D2"),
        (0.7, "Backward LSTM layer 1", "#C62828"),
        (2.7, "Forward LSTM layer 2", "#0D47A1"),
        (2.2, "Backward LSTM layer 2", "#B71C1C"),
        (4.0, "LM output head: linear + softmax", "#F9A825"),
    ]:
        fig_elmo.add_annotation(
            x=4.95, y=elmo_y_pos, text=elmo_label,
            showarrow=False, font=dict(size=10, color=elmo_col), xanchor="left"
        )

    fig_elmo.add_annotation(
        x=2.5, y=4.55,
        text="Pre-trained bidirectional language model",
        showarrow=False, font=dict(size=12)
    )

    fig_elmo.update_layout(
        title="ELMo Architecture: Bi-directional 2-Layer LSTM + Prediction Head",
        xaxis=dict(range=[0.3, 6.9], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.5, 4.8], showticklabels=False, showgrid=False, zeroline=False),
        height=430,
        margin=dict(t=55, b=10),
        plot_bgcolor="white",
    )
    fig_elmo
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ELMo: Contextual Disambiguation

    Because embeddings are *recomputed* for each sentence, the same token
    gets a different vector in different contexts — solving the polysemy problem.
    """)
    return


@app.cell
def _(mo):
    context_selector = mo.ui.dropdown(
        options=["bank office (financial)", "river bank (geographic)", "blood bank (medical)"],
        value="bank office (financial)",
        label="Select context for 'bank':",
    )
    context_selector
    return (context_selector,)


@app.cell
def _(context_selector, go, mo, np):
    context_embeddings = {
        "bank office (financial)": np.array([3.2, 3.8]),
        "river bank (geographic)": np.array([-3.4, 2.1]),
        "blood bank (medical)":    np.array([1.1, -2.9]),
    }
    context_colors = {
        "bank office (financial)": "#2196F3",
        "river bank (geographic)": "#4CAF50",
        "blood bank (medical)":    "#F44336",
    }

    selected = context_selector.value
    active_pt = context_embeddings[selected]
    active_col = context_colors[selected]

    fig_elmo_ctx = go.Figure()
    # Draw all three faintly
    for ctx, pt_e in context_embeddings.items():
        col_e = context_colors[ctx]
        opacity_e = 1.0 if ctx == selected else 0.15
        fig_elmo_ctx.add_trace(go.Scatter(
            x=[0, pt_e[0]], y=[0, pt_e[1]],
            mode="lines+markers+text",
            line=dict(color=col_e, width=3 if ctx == selected else 1),
            marker=dict(size=[4, 14], color=col_e, opacity=[opacity_e, opacity_e]),
            text=["", ctx],
            textposition="top center",
            textfont=dict(size=11, color=col_e),
            opacity=opacity_e,
            name=ctx,
        ))

    fig_elmo_ctx.update_layout(
        title=f'ELMo: "bank" embedding shifts with context — currently: {selected}',
        xaxis=dict(zeroline=True, range=[-5, 5]),
        yaxis=dict(zeroline=True, range=[-4, 5], scaleanchor="x"),
        height=380,
        margin=dict(t=60),
    )
    mo.vstack([
        fig_elmo_ctx,
        mo.md(f"""
        **Current vector for "bank" in context "{selected}":**
        `({active_pt[0]}, {active_pt[1]})`

        ELMo recomputes this vector from scratch for every sentence —
        so the same word gets a different position in every context.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ELMo Disadvantage

    > **ELMo is computationally very intensive.**
    > The recurrent LSTM passes are hard to parallelise because each step depends on the previous hidden state.

    → This is one reason **Transformers** replaced recurrence with self-attention and scale much better on modern hardware.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 7  Tokenisation

    Before any model can learn embeddings, text must be split into **tokens** —
    the atomic units that form the vocabulary.

    | Model | Token unit | OOV handling |
    |---|---|---|
    | **Word2Vec** | Full words | Unknown words get no vector |
    | **ELMo** | Characters | Can represent any word |
    | **BERT** | Subwords (WordPiece) | Split unknown words into subword pieces |
    | **GPT** | Subwords (BPE) | Split unknown words into byte-pair pieces |

    ### Subword Tokenisation Example
    """)
    return


@app.cell
def _(mo):
    tokenize_input = mo.ui.text(
        value="Huggingface is nontrivial!",
        label="Enter text to tokenise:",
        full_width=True,
    )
    tokenize_input
    return (tokenize_input,)


@app.cell
def _(mo, tokenize_input):
    text_in = tokenize_input.value

    # Simplified simulation of BERT WordPiece and GPT BPE for known examples
    # (Real tokenisers require the actual vocab — we simulate the lecture examples)
    def simulate_gpt_tokenizer(text):
        """Approximate BPE-style (character + common subwords)."""
        # Very rough simulation for demo purposes
        import re
        tokens = []
        for word in re.findall(r"\w+|[^\w\s]", text):
            if len(word) <= 3:
                tokens.append(word)
            elif word.lower() in ("huggingface",):
                tokens.extend(["H", "ugging", "face"])
            elif word.lower() in ("nontrivial",):
                tokens.extend(["non", "tr", "ivial"])
            elif word.lower() in ("is",):
                tokens.append(" is")
            else:
                # Split into trigrams as approximation
                tokens.extend([word[i:i+4] for i in range(0, len(word), 4)])
        return tokens

    def simulate_bert_tokenizer(text):
        """Approximate WordPiece-style (## prefix for subword continuations)."""
        import re
        tokens = []
        for word in re.findall(r"\w+|[^\w\s]", text):
            lower = word.lower()
            if lower in ("is",):
                tokens.append("is")
            elif lower in ("huggingface",):
                tokens.extend(["hugging", "##face"])
            elif lower in ("nontrivial",):
                tokens.extend(["non", "##tri", "##vial"])
            elif word in ("!",):
                tokens.append("!")
            else:
                tokens.extend([lower[i:i+4] if i==0 else "##" + lower[i:i+4]
                                for i in range(0, len(lower), 4)])
        return tokens

    gpt_tokens  = simulate_gpt_tokenizer(text_in)
    bert_tokens = simulate_bert_tokenizer(text_in)
    word_tokens = text_in.lower().split()

    mo.md(f"""
    #### Tokenisation of: *"{text_in}"*

    | Strategy | Tokens | Count |
    |---|---|---|
    | **Word-level** (Word2Vec) | `{word_tokens}` | {len(word_tokens)} |
    | **BPE** (GPT-style) | `{gpt_tokens}` | {len(gpt_tokens)} |
    | **WordPiece** (BERT-style) | `{bert_tokens}` | {len(bert_tokens)} |

    > Subword tokenisers handle *any* word by splitting into known pieces.
    > Words outside the vocab (OOV) are never truly "unknown" — they are
    > decomposed until all pieces are in the vocabulary.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 8  Summary

    | Concept | Key Idea |
    |---|---|
    | **Embeddings** | Continuous vector space; similar words are close together |
    | **Word2Vec** | Learn embeddings by predicting word context (Skip-gram / CBOW) |
    | **Cosine Similarity** | Angle between vectors; magnitude-invariant |
    | **Polysemy problem** | Word2Vec gives one static vector per word |
    | **ELMo** | Pre-trained bi-directional LSTM; dynamic embeddings per sentence |
    | **ELMo cost** | Computationally very intensive; recurrent training is hard to parallelise |
    | **Tokenisation** | Subword tokens allow handling of OOV words |
    | **Outlook** | Transformers replace LSTMs with self-attention for better parallelism |

    ---

    ## 9  Quick Knowledge Check
    """)
    return


@app.cell
def _(mo):
    q1 = mo.ui.radio(
        options={
            "It uses subword tokenisation": "wrong",
            "The same vector is used regardless of context": "correct",
            "It cannot handle large vocabularies": "wrong",
            "It requires labelled training data": "wrong",
        },
        label="**Q1.** What is the main disadvantage of Word2Vec embeddings?",
    )
    q1
    return (q1,)


@app.cell
def _(mo, q1):
    if q1.value == "correct":
        mo.callout(mo.md("**Correct!** Word2Vec assigns one fixed vector per word, so polysemous words like *bank* get a muddled average vector."), kind="success")
    elif q1.value == "wrong":
        mo.callout(mo.md("Not quite — think about what changes (or doesn't) when the same word appears in different sentences."), kind="warn")
    else:
        mo.md("")
    return


@app.cell
def _(mo):
    q2 = mo.ui.radio(
        options={
            "cos(90°) = 1": "wrong",
            "cos(0°) = 1": "correct",
            "cos(180°) = 0": "wrong",
            "cos(0°) = 0": "wrong",
        },
        label="**Q2.** Which statement about cosine similarity is correct?",
    )
    q2
    return (q2,)


@app.cell
def _(mo, q2):
    if q2.value == "correct":
        mo.callout(mo.md("**Correct!** When two vectors point in the exact same direction (angle = 0°), cos(0°) = 1 — maximum similarity."), kind="success")
    elif q2.value == "wrong":
        mo.callout(mo.md("Not quite — recall: cos(0°) = 1 (identical), cos(90°) = 0 (orthogonal), cos(180°) = −1 (opposite)."), kind="warn")
    else:
        mo.md("")
    return


@app.cell
def _(mo):
    q3 = mo.ui.radio(
        options={
            "Word2Vec with skip-gram": "wrong",
            "ELMo with bi-directional LSTM": "correct",
            "Bag-of-words with TF-IDF": "wrong",
            "GloVe word vectors": "wrong",
        },
        label='**Q3.** Which model produces *different* embeddings for "bank" in "river bank" vs "bank office"?',
    )
    q3
    return (q3,)


@app.cell
def _(mo, q3):
    if q3.value == "correct":
        mo.callout(mo.md("**Correct!** ELMo computes contextual embeddings — the same token gets a different vector depending on the surrounding sentence."), kind="success")
    elif q3.value == "wrong":
        mo.callout(mo.md("Not quite — which model in this lecture produces *dynamic* embeddings that change per sentence?"), kind="warn")
    else:
        mo.md("")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 10  Hands-On Exercise

    Use the interactive tools above to complete these tasks:

    1. **BoW limitation** — Start from the default movie documents in Section 1.
       Which two documents are closest in BoW space, and why does BoW still fail to capture that `great` and `amazing` are semantically similar?

    2. **Cosine similarity** — Use the calculator in Section 4 to verify:
       - *wonderful* = (5, 1) vs *adorable* = (5, 4): expected ≈ **0.89**
       - *wonderful* = (5, 1) vs *horrible* = (−5, 3): expected ≈ **−0.74**

    3. **Polysemy** — In Section 6, switch between the three "bank" contexts.
       How does the ELMo vector direction change compared to the static Word2Vec average?

    4. **Reflection** — Why does ELMo lead to better results than Word2Vec on tasks
       like sentiment analysis of ambiguous sentences? (Think about polysemy.)
    """)
    return


if __name__ == "__main__":
    app.run()
