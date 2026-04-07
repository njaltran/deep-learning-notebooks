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
    import re
    from collections import Counter

    import numpy as np
    import plotly.graph_objects as go
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    return (
        Counter,
        CountVectorizer,
        LogisticRegression,
        TfidfVectorizer,
        accuracy_score,
        go,
        math,
        np,
        re,
        train_test_split,
    )


@app.cell
def _(Counter, math, np, re):
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )

    def normalize_spaces(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def simple_word_tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)

    def clean_text(
        text: str,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_emojis: bool = False,
    ) -> str:
        cleaned = text
        if remove_html:
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        if remove_urls:
            cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
        if remove_emojis:
            cleaned = emoji_pattern.sub(" ", cleaned)
        if lowercase:
            cleaned = cleaned.lower()
        if remove_numbers:
            cleaned = re.sub(r"\d+", " ", cleaned)
        if remove_punctuation:
            cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        return normalize_spaces(cleaned)

    def tokenize_for_bow(text: str, lowercase: bool = True) -> list[str]:
        base = text.lower() if lowercase else text
        return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", base)

    def parse_documents(multiline_text: str) -> list[str]:
        return [line.strip() for line in multiline_text.split("\n") if line.strip()]

    def build_vocabulary(tokenized_docs: list[list[str]]) -> list[str]:
        vocabulary = []
        seen = set()
        for tokens in tokenized_docs:
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    vocabulary.append(token)
        return vocabulary

    def compute_bow_matrix(
        docs: list[str], scheme: str
    ) -> tuple[list[list[str]], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tokenized_docs = [tokenize_for_bow(doc) for doc in docs]
        tokenized_docs = [tokens for tokens in tokenized_docs if tokens]
        vocabulary = build_vocabulary(tokenized_docs)

        if not tokenized_docs or not vocabulary:
            empty_matrix = np.zeros((0, 0))
            empty_vector = np.array([])
            return (
                tokenized_docs,
                vocabulary,
                empty_matrix,
                empty_matrix,
                empty_vector,
                empty_vector,
                empty_vector,
            )

        counts = np.array(
            [
                [Counter(tokens).get(word, 0) for word in vocabulary]
                for tokens in tokenized_docs
            ],
            dtype=float,
        )
        doc_lengths = np.array([len(tokens) for tokens in tokenized_docs], dtype=float)
        doc_freq = np.count_nonzero(counts > 0, axis=0).astype(float)
        idf = np.where(doc_freq > 0, np.log10(len(tokenized_docs) / doc_freq), 0.0)

        if scheme == "One-hot encoding":
            matrix = (counts > 0).astype(float)
        elif scheme == "Absolute frequency":
            matrix = counts
        elif scheme == "Relative frequency":
            matrix = counts / doc_lengths[:, None]
        else:
            matrix = (counts / doc_lengths[:, None]) * idf[None, :]

        return tokenized_docs, vocabulary, matrix, counts, doc_lengths, doc_freq, idf

    def format_value(value: float) -> str:
        if math.isclose(value, round(value), abs_tol=1e-9):
            return str(int(round(value)))
        return f"{value:.2f}"

    def markdown_matrix(
        doc_labels: list[str], vocabulary: list[str], matrix: np.ndarray
    ) -> str:
        if not vocabulary:
            return "_No vocabulary extracted yet._"

        header = "| Doc | " + " | ".join(vocabulary) + " |"
        separator = "|---|" + "|".join(["---"] * len(vocabulary)) + "|"
        rows = []
        for label, row in zip(doc_labels, matrix):
            values = " | ".join(format_value(float(value)) for value in row)
            rows.append(f"| {label} | {values} |")
        return "\n".join([header, separator, *rows])

    def document_frequency_table(
        vocabulary: list[str], doc_freq: np.ndarray, idf: np.ndarray
    ) -> str:
        if not vocabulary:
            return "_No words to summarize._"

        header = "| Word | Docs with word | log10(N/df) |"
        separator = "|---|---:|---:|"
        rows = [
            f"| {word} | {int(freq)} | {value:.2f} |"
            for word, freq, value in zip(vocabulary, doc_freq, idf)
        ]
        return "\n".join([header, separator, *rows])

    return (
        build_vocabulary,
        clean_text,
        compute_bow_matrix,
        document_frequency_table,
        format_value,
        markdown_matrix,
        parse_documents,
        simple_word_tokenize,
        tokenize_for_bow,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 09 — Text Preprocessing & Bag-of-Words

    *Natural Language Processing Lab · Prof. Dr. Diana Hristova · 02.04.2026*

    ## Learning Objectives
    - Understand why text cleaning matters before analysis
    - See why classical machine learning needs structured numeric text features
    - Build Bag-of-Words representations with different weighting schemes
    - Use `min_df` and `max_df` to remove very common or very rare words
    - Recognize the main limitation of Bag-of-Words: it ignores context

    ## Connection to Classical ML
    Linear regression, logistic regression, trees, and SVMs expect a **design matrix**:
    each row is one observation and each column is one numeric feature.

    Raw text is not in that format. **Vectorization** converts text into fixed-length
    numeric feature vectors so the same ML tools you already know can be applied.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Motivation: Why Clean Text?

    Text data often contains noise:
    - HTML tags
    - URLs
    - emojis
    - inconsistent casing
    - punctuation and formatting artifacts

    The goal is not "remove everything unusual." The goal is **remove what hurts the
    chosen representation**. Modern LLMs can tolerate some noise, and emojis or hashtags
    may still carry useful meaning.

    ### Cleaning Playground
    """)
    return


@app.cell
def _(mo):
    noisy_text = mo.ui.text_area(
        value=(
            "<article><h2>Berliner Fernsehturm was AMAZING!!! 😍</h2>"
            "<p>Read more at https://visit.berlin/example. "
            "Ticket price: 25 EUR. #Berlin #Travel</p></article>"
        ),
        label="Noisy text example",
    )
    noisy_text
    return (noisy_text,)


@app.cell
def _(mo):
    lowercase_box = mo.ui.checkbox(value=True, label="Lowercase")
    remove_html_box = mo.ui.checkbox(value=True, label="Remove HTML tags")
    remove_urls_box = mo.ui.checkbox(value=True, label="Remove URLs")
    remove_punctuation_box = mo.ui.checkbox(value=True, label="Remove punctuation")
    remove_numbers_box = mo.ui.checkbox(value=False, label="Remove numbers")
    remove_emojis_box = mo.ui.checkbox(value=False, label="Remove emojis")

    mo.vstack(
        [
            mo.hstack(
                [lowercase_box, remove_html_box, remove_urls_box],
                justify="start",
                gap=1,
            ),
            mo.hstack(
                [remove_punctuation_box, remove_numbers_box, remove_emojis_box],
                justify="start",
                gap=1,
            ),
        ],
        gap=1,
    )
    return (
        lowercase_box,
        remove_emojis_box,
        remove_html_box,
        remove_numbers_box,
        remove_punctuation_box,
        remove_urls_box,
    )


@app.cell
def _(
    clean_text,
    lowercase_box,
    noisy_text,
    mo,
    re,
    remove_emojis_box,
    remove_html_box,
    remove_numbers_box,
    remove_punctuation_box,
    remove_urls_box,
    simple_word_tokenize,
):
    original = noisy_text.value.strip() or ""
    cleaned = clean_text(
        original,
        lowercase=lowercase_box.value,
        remove_html=remove_html_box.value,
        remove_urls=remove_urls_box.value,
        remove_punctuation=remove_punctuation_box.value,
        remove_numbers=remove_numbers_box.value,
        remove_emojis=remove_emojis_box.value,
    )

    original_tokens = simple_word_tokenize(original)
    cleaned_tokens = simple_word_tokenize(cleaned)

    removed_items = []
    if remove_html_box.value and re.search(r"<[^>]+>", original):
        removed_items.append("HTML")
    if remove_urls_box.value and re.search(r"https?://\S+|www\.\S+", original):
        removed_items.append("URLs")
    if remove_punctuation_box.value:
        removed_items.append("punctuation")
    if remove_numbers_box.value and re.search(r"\d+", original):
        removed_items.append("numbers")
    if remove_emojis_box.value:
        removed_items.append("emojis")

    removed_text = ", ".join(removed_items) if removed_items else "nothing explicit"
    token_preview = ", ".join(cleaned_tokens) if cleaned_tokens else "no word tokens"

    mo.md(
        f"""
        ### Cleaning Preview

        **Potentially removed:** {removed_text}

        **Original text**
        ```text
        {original}
        ```

        **Cleaned text**
        ```text
        {cleaned}
        ```

        **Word tokens before cleaning:** {len(original_tokens)}

        **Word tokens after cleaning:** {len(cleaned_tokens)}

        **Cleaned tokens:** {token_preview}

        The important modeling question is: *does this removal reduce noise, or did it also
        remove signal?* For sentiment analysis, for example, `😍` or `!!!` can still matter.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. From Unstructured Text to Structured Vectors

    Consider this review:

    > Berliner Fernsehturm has a lot of history and is definitely worth a visit.

    A human can read this directly. A classical ML model cannot. It needs a fixed number
    of columns, such as:

    | doc_id | word_1 | word_2 | word_3 | ... |
    |---|---:|---:|---:|---:|
    | 1 | 0 | 2 | 1 | ... |

    Bag-of-Words is the simplest way to build this representation:
    - build a **dictionary / vocabulary** from the corpus
    - assign one feature per vocabulary word
    - fill each feature with a score such as binary presence, count, relative frequency,
      or TF-IDF

    ### Bag-of-Words Explorer
    """)
    return


@app.cell
def _(mo):
    bow_docs_input = mo.ui.text_area(
        value="movie was great\nmovie was great was amazing\nmovie was bad",
        label="Enter one document per line",
    )
    bow_scheme = mo.ui.dropdown(
        options=[
            "One-hot encoding",
            "Absolute frequency",
            "Relative frequency",
            "TF-IDF (slide formula)",
        ],
        value="One-hot encoding",
        label="Feature value strategy",
    )
    mo.hstack([bow_docs_input, bow_scheme], justify="center", gap=1)
    return bow_docs_input, bow_scheme


@app.cell
def _(bow_docs_input, bow_scheme, compute_bow_matrix, parse_documents):
    bow_docs = parse_documents(bow_docs_input.value)
    (
        bow_tokenized_docs,
        bow_vocabulary,
        bow_matrix,
        bow_counts,
        bow_doc_lengths,
        bow_doc_freq,
        bow_idf,
    ) = compute_bow_matrix(bow_docs, bow_scheme.value)
    bow_doc_labels = [f"Doc{i}" for i in range(1, len(bow_tokenized_docs) + 1)]
    return (
        bow_counts,
        bow_doc_freq,
        bow_doc_labels,
        bow_doc_lengths,
        bow_docs,
        bow_idf,
        bow_matrix,
        bow_tokenized_docs,
        bow_vocabulary,
    )


@app.cell
def _(
    bow_doc_freq,
    bow_doc_labels,
    bow_doc_lengths,
    bow_idf,
    bow_matrix,
    bow_scheme,
    bow_vocabulary,
    document_frequency_table,
    markdown_matrix,
    mo,
):
    if not bow_vocabulary:
        _output = mo.md("_Add at least one document with alphabetic tokens._")
    else:
        _advantages = {
            "One-hot encoding": "Easy to calculate and interpret; useful for short texts.",
            "Absolute frequency": "Reflects how often a word appears, so repeated words matter more.",
            "Relative frequency": "Accounts for document length, so short and long documents become more comparable.",
            "TF-IDF (slide formula)": "Downweights words that appear in almost every document and highlights more distinctive words.",
        }
        _disadvantages = {
            "One-hot encoding": "A word that appears once is treated the same as a word that appears many times.",
            "Absolute frequency": "Longer documents naturally get larger counts, so documents are less comparable.",
            "Relative frequency": "Very common corpus-wide words can still dominate the representation.",
            "TF-IDF (slide formula)": "Harder to interpret directly, and values depend on the full corpus.",
        }

        _doc_lengths_text = ", ".join(
            f"{label}: {int(length)} tokens"
            for label, length in zip(bow_doc_labels, bow_doc_lengths)
        )

        _extra = ""
        if bow_scheme.value == "TF-IDF (slide formula)":
            _extra = (
                "\n\n**Lecture-matching TF-IDF:** "
                "`relative_frequency(word, doc) * log10(N / df(word))`\n\n"
                + document_frequency_table(bow_vocabulary, bow_doc_freq, bow_idf)
            )

        _output = mo.md(
            f"""
            **Vocabulary ({len(bow_vocabulary)} words):** {", ".join(bow_vocabulary)}

            **Document lengths:** {_doc_lengths_text}

            {markdown_matrix(bow_doc_labels, bow_vocabulary, bow_matrix)}

            **Advantage:** {_advantages[bow_scheme.value]}

            **Disadvantage:** {_disadvantages[bow_scheme.value]}
            {_extra}
            """
        )
    _output
    return


@app.cell
def _(bow_doc_labels, bow_matrix, bow_scheme, bow_vocabulary, format_value, go, mo):
    if bow_vocabulary:
        _text_values = [
            [format_value(float(value)) for value in row]
            for row in bow_matrix
        ]

        _fig = go.Figure(
            data=go.Heatmap(
                z=bow_matrix,
                x=bow_vocabulary,
                y=bow_doc_labels,
                colorscale="Blues",
                text=_text_values,
                texttemplate="%{text}",
                hovertemplate="Document=%{y}<br>Word=%{x}<br>Value=%{z:.3f}<extra></extra>",
            )
        )
        _fig.update_layout(
            template="plotly_dark",
            height=360,
            title=f"{bow_scheme.value} document-term matrix",
            xaxis_title="Vocabulary word",
            yaxis_title="Document",
            margin=dict(l=50, r=30, t=60, b=50),
        )
        _output = _fig
    else:
        _output = mo.md("")
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Feature Selection with `min_df` and `max_df`

    Structured text data still needs feature selection. One simple rule is:
    - remove words that appear in **too few** documents (`min_df`)
    - remove words that appear in **too many** documents (`max_df`)

    This reduces noise and can improve downstream models.

    ### Document Frequency Filter
    """)
    return


@app.cell
def _(mo):
    filter_corpus_input = mo.ui.text_area(
        value=(
            "news about berlin economy and markets\n"
            "news about berlin transport and traffic\n"
            "movie review with great acting and story\n"
            "movie review with bad acting and pacing\n"
            "travel guide to berlin tower and museum\n"
            "travel guide to berlin food and markets"
        ),
        label="Corpus for document-frequency filtering (one document per line)",
    )
    filter_corpus_input
    return (filter_corpus_input,)


@app.cell
def _(filter_corpus_input, mo, parse_documents):
    filter_docs = parse_documents(filter_corpus_input.value)
    n_docs = max(len(filter_docs), 1)

    min_df_slider = mo.ui.slider(
        start=1,
        stop=n_docs,
        step=1,
        value=min(2, n_docs),
        label="min_df (minimum number of documents)",
    )
    max_df_slider = mo.ui.slider(
        start=1,
        stop=n_docs,
        step=1,
        value=n_docs,
        label="max_df (maximum number of documents)",
    )
    mo.hstack([min_df_slider, max_df_slider], justify="center", gap=1.5)
    return filter_docs, max_df_slider, min_df_slider


@app.cell
def _(
    build_vocabulary,
    filter_docs,
    max_df_slider,
    min_df_slider,
    mo,
    tokenize_for_bow,
):
    tokenized_filter_docs = [tokenize_for_bow(doc) for doc in filter_docs if doc.strip()]
    filter_vocabulary = build_vocabulary(tokenized_filter_docs)
    filter_doc_freq = None
    filter_vocabulary_output = None
    kept_words = None
    removed_common = None

    if min_df_slider.value > max_df_slider.value:
        _output = mo.md("Set `min_df <= max_df` to keep a valid vocabulary.")
    elif not filter_vocabulary:
        _output = mo.md("_No vocabulary extracted from the corpus._")
    else:
        filter_doc_freq = {
            _word: sum(1 for tokens in tokenized_filter_docs if _word in set(tokens))
            for _word in filter_vocabulary
        }

        kept_words = [
            _word
            for _word in filter_vocabulary
            if min_df_slider.value <= filter_doc_freq[_word] <= max_df_slider.value
        ]
        removed_rare = [
            _word
            for _word in filter_vocabulary
            if filter_doc_freq[_word] < min_df_slider.value
        ]
        removed_common = [
            _word
            for _word in filter_vocabulary
            if filter_doc_freq[_word] > max_df_slider.value
        ]

        _rows = []
        for _word in filter_vocabulary:
            if _word in kept_words:
                _status = "kept"
            elif _word in removed_rare:
                _status = "removed as too rare"
            else:
                _status = "removed as too common"
            _rows.append(f"| {_word} | {filter_doc_freq[_word]} | {_status} |")

        _summary_table = "\n".join(
            [
                "| Word | Docs with word | Status |",
                "|---|---:|---|",
                *_rows,
            ]
        )

        _kept_text = ", ".join(kept_words) if kept_words else "none"
        _removed_rare_text = ", ".join(removed_rare) if removed_rare else "none"
        _removed_common_text = ", ".join(removed_common) if removed_common else "none"

        _output = mo.md(
            f"""
            **Kept vocabulary:** {_kept_text}

            **Removed as too rare:** {_removed_rare_text}

            **Removed as too common:** {_removed_common_text}

            {_summary_table}
            """
        )
        filter_vocabulary_output = filter_vocabulary

    _output
    return filter_doc_freq, filter_vocabulary_output, kept_words, removed_common


@app.cell
def _(filter_doc_freq, filter_vocabulary, go, kept_words, mo, removed_common):
    if filter_doc_freq is not None and filter_vocabulary is not None:
        _colors = []
        for _word in filter_vocabulary:
            if _word in kept_words:
                _colors.append("#00CC96")
            elif _word in removed_common:
                _colors.append("#FFA15A")
            else:
                _colors.append("#EF553B")

        _fig = go.Figure(
            data=go.Bar(
                x=filter_vocabulary,
                y=[filter_doc_freq[_word] for _word in filter_vocabulary],
                marker_color=_colors,
            )
        )
        _fig.update_layout(
            template="plotly_dark",
            height=320,
            title="Document frequency per word",
            xaxis_title="Word",
            yaxis_title="Number of documents",
            margin=dict(l=50, r=30, t=60, b=50),
        )
        _output = _fig
    else:
        _output = mo.md("")
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Main Limitation: Bag-of-Words Ignores Context

    Bag-of-Words treats each word in isolation. Word order is ignored.

    That means the following two documents can get the **same representation** even
    though their meanings are very different.

    ### Context Loss Demo
    """)
    return


@app.cell
def _(mo):
    context_doc_a = mo.ui.text_area(
        value="The meal is horrible, but the service is great.",
        label="Document A",
    )
    context_doc_b = mo.ui.text_area(
        value="The service is horrible, but the meal is great.",
        label="Document B",
    )
    mo.hstack([context_doc_a, context_doc_b], justify="center", gap=1)
    return context_doc_a, context_doc_b


@app.cell
def _(
    build_vocabulary,
    context_doc_a,
    context_doc_b,
    markdown_matrix,
    mo,
    np,
    tokenize_for_bow,
):
    context_tokens = [
        tokenize_for_bow(context_doc_a.value),
        tokenize_for_bow(context_doc_b.value),
    ]
    context_vocab = build_vocabulary(context_tokens)

    if not context_vocab:
        _output = mo.md("_Enter some text in both boxes._")
    else:
        _context_counts = np.array(
            [
                [tokens.count(_word) for _word in context_vocab]
                for tokens in context_tokens
            ],
            dtype=float,
        )
        _identical_vectors = np.array_equal(_context_counts[0], _context_counts[1])
        _same_text = "Yes" if _identical_vectors else "No"

        _output = mo.md(
            f"""
            {markdown_matrix(["DocA", "DocB"], context_vocab, _context_counts)}

            **Same Bag-of-Words representation?** {_same_text}

            If the answer is `Yes`, Bag-of-Words cannot distinguish the two meanings because
            it only sees **which words occur** and **how often**.
            """
        )
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Applying Bag-of-Words for Decision Support

    Once the text is vectorized, we can plug it into a standard classifier.
    Below, a small sentiment model uses either binary counts, raw counts, or TF-IDF.

    **Note:** `scikit-learn` uses a slightly different TF-IDF implementation than the
    lecture formula above (smoothed IDF). That is normal; the slide deck itself notes
    that packages differ.

    ### Mini Sentiment Classifier
    """)
    return


@app.cell
def _(mo):
    classifier_scheme = mo.ui.dropdown(
        options=[
            "Binary counts",
            "Absolute counts",
            "TF-IDF",
        ],
        value="TF-IDF",
        label="Vectorizer",
    )
    classifier_min_df = mo.ui.slider(
        start=1,
        stop=3,
        step=1,
        value=1,
        label="min_df (documents)",
    )
    classifier_max_df = mo.ui.slider(
        start=0.30,
        stop=1.00,
        step=0.05,
        value=1.00,
        label="max_df (proportion of documents)",
    )
    mo.hstack(
        [classifier_scheme, classifier_min_df, classifier_max_df],
        justify="center",
        gap=1,
    )
    return classifier_max_df, classifier_min_df, classifier_scheme


@app.cell
def _():
    review_texts = [
        "the movie was great and visually stunning",
        "an amazing performance and a touching story",
        "smart script with excellent acting and pacing",
        "the film was funny warm and surprisingly moving",
        "great soundtrack and strong characters",
        "brilliant direction and memorable scenes",
        "the story was engaging and the ending was satisfying",
        "excellent movie with sharp dialogue and energy",
        "a beautiful film with strong performances",
        "heartfelt and entertaining with a great cast",
        "the movie was boring and painfully slow",
        "bad acting and a confusing script",
        "the film was dull predictable and too long",
        "terrible pacing with weak characters",
        "boring story and disappointing ending",
        "awful movie with flat jokes and messy editing",
        "the plot was bad and the acting was worse",
        "poor direction and forgettable scenes",
        "a lifeless film with weak performances",
        "frustrating and noisy with no emotional impact",
    ]
    review_labels = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    return review_labels, review_texts


@app.cell
def _(
    CountVectorizer,
    LogisticRegression,
    TfidfVectorizer,
    accuracy_score,
    classifier_max_df,
    classifier_min_df,
    classifier_scheme,
    review_labels,
    review_texts,
    train_test_split,
):
    X_train, X_test, y_train, y_test = train_test_split(
        review_texts,
        review_labels,
        test_size=0.25,
        random_state=7,
        stratify=review_labels,
    )

    try:
        if classifier_scheme.value == "Binary counts":
            vectorizer = CountVectorizer(
                binary=True,
                min_df=classifier_min_df.value,
                max_df=classifier_max_df.value,
            )
        elif classifier_scheme.value == "Absolute counts":
            vectorizer = CountVectorizer(
                binary=False,
                min_df=classifier_min_df.value,
                max_df=classifier_max_df.value,
            )
        else:
            vectorizer = TfidfVectorizer(
                min_df=classifier_min_df.value,
                max_df=classifier_max_df.value,
            )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(max_iter=2000, random_state=7)
        model.fit(X_train_vec, y_train)

        train_pred = model.predict(X_train_vec)
        test_pred = model.predict(X_test_vec)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]

        n_top = min(5, len(feature_names))
        neg_idx = list(coefficients.argsort()[:n_top])
        pos_candidates = list(coefficients.argsort()[::-1])
        pos_idx = []
        for idx in pos_candidates:
            if idx not in neg_idx:
                pos_idx.append(int(idx))
            if len(pos_idx) == n_top:
                break

        test_examples = list(zip(X_test, y_test, test_pred))
        classifier_error = None
    except ValueError as exc:
        X_train_vec = None
        X_test_vec = None
        model = None
        train_accuracy = None
        test_accuracy = None
        feature_names = None
        coefficients = None
        neg_idx = None
        pos_idx = None
        test_examples = None
        classifier_error = str(exc)

    return (
        X_test_vec,
        X_train_vec,
        classifier_error,
        coefficients,
        feature_names,
        neg_idx,
        pos_idx,
        test_accuracy,
        test_examples,
        train_accuracy,
    )


@app.cell
def _(
    X_train_vec,
    classifier_error,
    classifier_scheme,
    feature_names,
    mo,
    test_accuracy,
    test_examples,
    train_accuracy,
):
    if classifier_error:
        _output = mo.md(
            f"""
            The current `min_df` / `max_df` settings remove too much vocabulary for the
            classifier to train.

            ```text
            {classifier_error}
            ```
            """
        )
    else:
        _rows = [
            "| Review | True label | Predicted label |",
            "|---|---|---|",
        ]
        for text, truth, pred in test_examples:
            _truth_label = "positive" if truth == 1 else "negative"
            _pred_label = "positive" if pred == 1 else "negative"
            _rows.append(f"| {text} | {_truth_label} | {_pred_label} |")

        _results_table = "\n".join(_rows)

        _output = mo.md(
            f"""
            **Vectorizer:** {classifier_scheme.value}

            **Vocabulary size after filtering:** {len(feature_names)}

            **Train accuracy:** {train_accuracy:.2f}

            **Test accuracy:** {test_accuracy:.2f}

            **Training matrix shape:** {X_train_vec.shape[0]} documents x {X_train_vec.shape[1]} features

            {_results_table}
            """
        )
    _output
    return


@app.cell
def _(
    classifier_error,
    coefficients,
    feature_names,
    go,
    mo,
    neg_idx,
    pos_idx,
):
    if not classifier_error and feature_names is not None and coefficients is not None:
        _selected = neg_idx + pos_idx
        _words = [feature_names[i] for i in _selected]
        _values = [float(coefficients[i]) for i in _selected]
        _colors = ["#EF553B"] * len(neg_idx) + ["#00CC96"] * len(pos_idx)

        _fig = go.Figure(
            data=go.Bar(
                x=_values,
                y=_words,
                orientation="h",
                marker_color=_colors,
            )
        )
        _fig.update_layout(
            template="plotly_dark",
            height=360,
            title="Most negative vs most positive signal words",
            xaxis_title="Logistic regression coefficient",
            yaxis_title="Word",
            margin=dict(l=50, r=30, t=60, b=50),
        )
        _output = _fig
    else:
        _output = mo.md("")
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Concept | What You Learned |
    |---------|-----------------|
    | **Cleaning** | Remove or normalize noisy text parts when they hurt the chosen representation |
    | **Vectorization** | Convert text into fixed-length numeric feature vectors |
    | **Bag-of-Words** | Use vocabulary words as features and fill them with binary, count, relative, or TF-IDF values |
    | **Feature selection** | `min_df` and `max_df` remove words that are too rare or too common |
    | **Limitation** | Bag-of-Words ignores word order and therefore misses context |

    ### Key Takeaway
    Bag-of-Words is a strong baseline because it is simple, fast, and often effective.
    But it treats words independently, so it cannot fully represent meaning.

    ### Exercises
    1. Change the cleaning rules and decide which operations actually help for sentiment analysis.
    2. Compare one-hot, counts, relative frequency, and TF-IDF on your own tiny corpus.
    3. Adjust `min_df` and `max_df` until the classifier fails, then explain why.
    4. Edit the two context-loss documents until they no longer have the same Bag-of-Words vector.

    **Outlook:** Embeddings and transformer-based representations keep much more context
    than Bag-of-Words.
    """)
    return


if __name__ == "__main__":
    app.run()
