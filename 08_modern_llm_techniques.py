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
    # 08 — Modern LLM Techniques

    ## Learning Objectives
    - Understand fine-tuning vs training from scratch
    - See how LoRA makes fine-tuning parameter-efficient
    - Explore text embeddings and vector similarity
    - Build a simple RAG (Retrieval-Augmented Generation) pipeline
    - Overview of RLHF (Reinforcement Learning from Human Feedback)

    ## Connection to Classical ML
    Fine-tuning is **transfer learning** — you don't train from scratch, you adapt a
    pre-trained model to your task. Like using pre-trained word embeddings in NLP or
    pre-trained feature extractors in computer vision. LoRA makes this efficient by
    only updating a **tiny fraction** of parameters.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Fine-Tuning: Adapting Pre-Trained Models

    Instead of training a model from scratch (which requires massive data and compute),
    we can **fine-tune** a pre-trained model on our specific task.

    ```
    Pre-trained Model (general knowledge)
        ↓ Fine-tune on task-specific data
    Fine-tuned Model (specialized)
    ```

    **The problem:** Full fine-tuning updates ALL parameters. For a 7B parameter model,
    that means storing 7 billion gradient values — very memory-intensive.

    **The solution:** Parameter-efficient fine-tuning methods like **LoRA**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. LoRA: Low-Rank Adaptation

    LoRA freezes the original weights and adds small **low-rank** matrices:

    $$W' = W + \Delta W = W + BA$$

    where:
    - $W$: Original frozen weights $(d \times d)$
    - $B$: Low-rank matrix $(d \times r)$
    - $A$: Low-rank matrix $(r \times d)$
    - $r$: Rank (typically 4-64, much smaller than $d$)

    **Only $B$ and $A$ are trained** — the original model stays frozen.

    ### Parameter Savings Calculator
    """)
    return


@app.cell
def _(mo):
    lora_d_model = mo.ui.slider(
        start=256, stop=4096, step=256, value=1024, label="d_model (hidden size)"
    )
    lora_n_layers = mo.ui.slider(
        start=4, stop=48, step=4, value=12, label="Number of Layers"
    )
    lora_rank = mo.ui.slider(
        start=1, stop=64, step=1, value=8, label="LoRA Rank (r)"
    )
    mo.hstack([lora_d_model, lora_n_layers, lora_rank], justify="center", gap=1)
    return lora_d_model, lora_n_layers, lora_rank


@app.cell
def _(lora_d_model, lora_n_layers, lora_rank, mo):
    import plotly.graph_objects as _go

    d = lora_d_model.value
    n_layers = lora_n_layers.value
    r = lora_rank.value

    # Full fine-tuning: all attention weights (Q, K, V, O projections)
    params_per_layer_full = 4 * d * d  # Q, K, V, O
    total_full = params_per_layer_full * n_layers

    # LoRA: only B and A matrices for Q and V (typical configuration)
    params_per_layer_lora = 2 * (d * r + r * d)  # B and A for Q and V
    total_lora = params_per_layer_lora * n_layers

    ratio = total_lora / total_full * 100

    # Visualization
    _fig = _go.Figure()
    _fig.add_trace(_go.Bar(
        x=["Full Fine-tuning", "LoRA"],
        y=[total_full, total_lora],
        text=[f"{total_full:,.0f}", f"{total_lora:,.0f}"],
        textposition="auto",
        marker_color=["#EF553B", "#00CC96"],
    ))

    _fig.update_layout(
        template="plotly_dark",
        title=f"Trainable Parameters: LoRA saves {100-ratio:.1f}%",
        yaxis_title="Parameters",
        yaxis_type="log",
        height=350,
        margin=dict(l=50, r=30, t=60, b=50),
    )

    mo.md(
        f"""
        ### LoRA Parameter Comparison

        | | Full Fine-tuning | LoRA (rank={r}) |
        |---|---|---|
        | **Trainable params** | {total_full:,} | {total_lora:,} |
        | **% of original** | 100% | {ratio:.2f}% |
        | **Memory savings** | — | {100-ratio:.1f}% reduction |

        **How it works mathematically:**
        - Original weight matrix $W$: {d} x {d} = {d*d:,} parameters
        - LoRA matrices $B$({d} x {r}) + $A$({r} x {d}) = {d*r + r*d:,} parameters
        - Compression ratio: **{d*d / (d*r + r*d):.0f}x** fewer parameters per matrix

        At rank {r}, you're saying: "The task-specific update can be captured in a
        {r}-dimensional subspace of the full {d}-dimensional weight space."
        """
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Text Embeddings: Mapping Text to Vectors

    Embeddings convert text into dense vectors where **semantic similarity** corresponds
    to **vector proximity**. Similar sentences end up near each other in embedding space.

    $$\text{similarity}(a, b) = \cos(\theta) = \frac{a \cdot b}{\|a\| \|b\|}$$

    Cosine similarity ranges from -1 (opposite) to 1 (identical).

    ### Embedding Space Explorer

    Enter sentences below to see how they cluster in embedding space:
    """)
    return


@app.cell
def _(mo):
    embed_sentences = mo.ui.text_area(
        value="The cat sat on the mat\nThe dog lay on the rug\nMachine learning is fascinating\nDeep learning uses neural networks\nI love pizza\nPizza is my favorite food",
        label="Enter sentences (one per line)",
    )
    embed_sentences
    return (embed_sentences,)


@app.cell
def _(embed_sentences, mo):
    import numpy as _np
    import plotly.graph_objects as _go
    from sklearn.decomposition import PCA

    sentences = [s.strip() for s in (embed_sentences.value or "hello\nworld").split("\n") if s.strip()]

    # Simple bag-of-words embedding (to avoid needing sentence-transformers for quick demo)
    # Build vocabulary
    _all_words = set()
    for s in sentences:
        _all_words.update(s.lower().split())
    _vocab = sorted(_all_words)
    _word_to_idx = {w: i for i, w in enumerate(_vocab)}

    # TF-IDF-like embeddings
    embeddings = _np.zeros((len(sentences), len(_vocab)))
    for _i, s in enumerate(sentences):
        words = s.lower().split()
        for w in words:
            embeddings[_i, _word_to_idx[w]] += 1
        embeddings[_i] /= len(words) + 1e-8  # normalize by length

    # Compute cosine similarity matrix
    norms = _np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalized = embeddings / norms
    sim_matrix = normalized @ normalized.T

    # PCA to 2D for visualization
    if len(sentences) > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
    else:
        coords = embeddings[:, :2]

    from plotly.subplots import make_subplots as _make_subplots

    _fig = _make_subplots(
        rows=1, cols=2,
        subplot_titles=["Embedding Space (PCA 2D)", "Cosine Similarity Matrix"],
        column_widths=[0.5, 0.5],
    )

    # Scatter plot
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
              "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

    for _i, s in enumerate(sentences):
        short_label = s[:20] + "..." if len(s) > 20 else s
        _fig.add_trace(
            _go.Scatter(
                x=[coords[_i, 0]], y=[coords[_i, 1]],
                mode="markers+text",
                marker=dict(size=12, color=colors[_i % len(colors)]),
                text=[short_label],
                textposition="top center",
                textfont=dict(size=9),
                name=short_label,
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Similarity heatmap
    short_labels = [s[:15] + ".." if len(s) > 15 else s for s in sentences]
    _fig.add_trace(
        _go.Heatmap(
            z=sim_matrix,
            x=short_labels, y=short_labels,
            colorscale="Viridis",
            text=_np.round(sim_matrix, 2).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=9),
            showscale=False,
        ),
        row=1, col=2,
    )
    _fig.update_yaxes(autorange="reversed", row=1, col=2)

    _fig.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=50, r=30, t=60, b=80),
    )

    mo.md(
        """
        **What to notice:**
        - Semantically similar sentences cluster together in the 2D projection
        - The similarity matrix shows which pairs are most/least similar
        - This uses simple bag-of-words — real models (sentence-transformers) capture
          much richer semantic relationships

        *In production, you'd use a model like `all-MiniLM-L6-v2` from sentence-transformers
        for much better embeddings.*
        """
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. RAG: Retrieval-Augmented Generation

    RAG combines **retrieval** (finding relevant documents) with **generation** (LLM response).
    Instead of relying solely on what the model memorized during training, RAG lets it
    access external knowledge at inference time.

    ### The RAG Pipeline

    ```
    User Query
        ↓
    1. EMBED the query
        ↓
    2. RETRIEVE top-k similar documents from a vector store
        ↓
    3. AUGMENT the prompt with retrieved context
        ↓
    4. GENERATE answer using the LLM with context
    ```

    ### Interactive RAG Step-Through
    """)
    return


@app.cell
def _(mo):
    rag_query = mo.ui.text(
        value="What is backpropagation?",
        label="Your question",
    )
    rag_top_k = mo.ui.slider(
        start=1, stop=5, step=1, value=3, label="Top-k documents to retrieve"
    )
    rag_step = mo.ui.slider(
        start=1, stop=4, step=1, value=1, label="Pipeline Step"
    )
    mo.hstack([rag_query, rag_top_k, rag_step], justify="center", gap=1)
    return rag_query, rag_step, rag_top_k


@app.cell
def _(mo, rag_query, rag_step, rag_top_k, vocab):
    import numpy as _np

    query = rag_query.value or "What is backpropagation?"
    k = rag_top_k.value
    step = rag_step.value

    # Knowledge base (simulating document chunks)
    documents = [
        "Backpropagation is the algorithm for computing gradients in neural networks. It applies the chain rule of calculus layer by layer, from output to input, to determine how each weight should be adjusted.",
        "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction that reduces the loss function. The learning rate controls the step size.",
        "Transformers are a neural network architecture based on self-attention mechanisms. They process all positions in parallel, unlike RNNs which process sequentially.",
        "Convolutional Neural Networks use learned filters to detect spatial patterns in images. Each filter acts as a feature detector for edges, textures, or shapes.",
        "Loss functions measure the difference between predicted and actual values. Common choices include MSE for regression and cross-entropy for classification.",
        "LSTM networks solve the vanishing gradient problem by using a cell state with gates that control information flow. The forget gate determines what to discard.",
        "Attention mechanisms allow models to focus on relevant parts of the input. The query-key-value framework enables dynamic weighted combinations of information.",
        "Tokenization converts text into numerical tokens. BPE (Byte Pair Encoding) creates subword units that balance vocabulary size and sequence length.",
    ]

    # Simple TF-IDF-like embedding
    _all_words = set()
    for doc in documents + [query]:
        _all_words.update(doc.lower().split())
    _vocab = sorted(_all_words)
    _word_to_idx = {w: i for i, w in enumerate(_vocab)}

    def embed(text):
        words = text.lower().split()
        vec = _np.zeros(len(_vocab))
        for w in words:
            if w in _word_to_idx:
                vec[_word_to_idx[w]] += 1
        return vec / (_np.linalg.norm(vec) + 1e-8)

    query_emb = embed(query)
    doc_embs = _np.array([embed(d) for d in documents])

    # Compute similarities
    similarities = doc_embs @ query_emb
    top_indices = _np.argsort(similarities)[::-1][:k]
    top_docs = [documents[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]

    # Build step-by-step output
    steps_content = {
        1: f"""
        ### Step 1: EMBED the Query

        **Query:** "{query}"

        Convert to a vector representation (embedding):
        `query_embedding = embed("{query}")`

        Embedding dimension: {len(vocab)} (vocabulary-based)

        The query vector captures the semantic meaning — words like "backpropagation"
        and "gradient" will have non-zero values.
        """,
        2: f"""
        ### Step 2: RETRIEVE Top-{k} Documents

        **Query:** "{query}"

        Compute cosine similarity between query and all {len(documents)} documents:

        | Rank | Score | Document Preview |
        |------|-------|-----------------|
        """ + "\n".join(
            f"| {_i+1} | {top_scores[_i]:.3f} | {top_docs[_i][:80]}... |"
            for _i in range(k)
        ) + f"""

        Retrieved the {k} most relevant chunks from our knowledge base.
        """,
        3: """
        ### Step 3: AUGMENT the Prompt

        Combine the retrieved documents with the user's question:

        ```
        System: Answer the user's question using ONLY the context provided below.

        Context:
        """ + "\n".join(f"[{_i+1}] {top_docs[_i]}" for _i in range(k)) + f"""

        Question: {query}
        Answer:
        ```

        The LLM now has relevant context it wasn't trained on — this is the power of RAG.
        """,
        4: f"""
        ### Step 4: GENERATE Answer

        The LLM processes the augmented prompt and generates a grounded answer.

        **Without RAG:** The model relies only on training data (may hallucinate or be outdated).

        **With RAG:** The model answers based on retrieved evidence (more accurate, verifiable).

        **Key benefits of RAG:**
        - No need to fine-tune the model for new information
        - Answers are grounded in source documents (can cite sources)
        - Knowledge base can be updated without retraining
        - Reduces hallucination by providing relevant context
        """,
    }

    mo.md(steps_content[step])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. RLHF: Training LLMs to Follow Instructions

    RLHF (Reinforcement Learning from Human Feedback) is how raw LLMs become helpful
    assistants. The process has three stages:

    ### Stage 1: Pre-training (what you built in Notebook 07)
    Train on massive text data to predict the next token. The model learns language
    but has no concept of "helpful" or "harmful."

    ### Stage 2: Supervised Fine-Tuning (SFT)
    Fine-tune on high-quality (prompt, response) pairs written by humans.
    The model learns to follow instructions.

    ### Stage 3: RLHF
    1. **Train a Reward Model:** Human evaluators rank multiple model outputs
       for the same prompt. A reward model learns to predict human preferences.
    2. **PPO Training:** Use the reward model as a signal to fine-tune the LLM
       with reinforcement learning (specifically PPO — Proximal Policy Optimization).

    ```
    Pre-training    →    SFT           →    RLHF
    (next token)         (follow          (be helpful,
                          instructions)     harmless,
                                           honest)
    ```

    The reward model essentially says: "humans prefer this style of response."
    The LLM then learns to generate responses that score highly.

    ---

    ## 6. Putting It All Together

    Here's how modern LLM systems combine these techniques:

    | Technique | Purpose | When to Use |
    |-----------|---------|-------------|
    | **Pre-training** | General language understanding | Once (very expensive) |
    | **Fine-tuning** | Adapt to a specific task/domain | When you have task-specific data |
    | **LoRA** | Efficient fine-tuning | When full fine-tuning is too expensive |
    | **Prompt engineering** | Steer behavior without training | Always (cheapest option) |
    | **RAG** | Ground responses in external knowledge | When knowledge changes or is specialized |
    | **RLHF** | Align with human preferences | For assistant-style models |

    ---

    ## Summary

    | Concept | What You Learned |
    |---------|-----------------|
    | **Fine-tuning** | Adapt a pre-trained model to a specific task |
    | **LoRA** | Low-rank updates — train <1% of parameters for similar quality |
    | **Embeddings** | Dense vectors where semantic similarity = vector proximity |
    | **RAG** | Retrieve → Augment → Generate — ground LLM answers in real data |
    | **RLHF** | Align LLMs with human preferences using reward models + RL |

    ### Key Takeaway
    Modern LLMs are built in layers: pre-training gives language understanding,
    fine-tuning adds task capability, RLHF adds alignment, and RAG adds knowledge.
    You don't need all of these — choose based on your use case and budget.

    ---

    **Congratulations!** You've completed the full journey from single neurons to modern
    LLM systems. The concepts in these 8 notebooks are the foundation of everything
    happening in AI today.
    """)
    return


if __name__ == "__main__":
    app.run()
