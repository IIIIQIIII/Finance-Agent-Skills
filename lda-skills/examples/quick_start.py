"""
Quick Start Example: LDA Topic Modeling on Financial News

This example demonstrates the basic workflow for both Scikit-learn and Gensim
approaches to LDA topic modeling on financial documents.
"""

# Sample financial news articles
SAMPLE_DOCUMENTS = [
    """The Federal Reserve announced a 25 basis point interest rate hike to combat
    rising inflation. Economists predict further monetary policy tightening in the
    coming months as inflation remains above the central bank's 2% target.""",

    """Tech stocks rallied on strong earnings reports from major technology companies.
    Apple and Microsoft led the gains with impressive revenue growth driven by cloud
    computing and AI products.""",

    """Oil prices surged to multi-month highs amid supply concerns and geopolitical
    tensions in the Middle East. Energy sector stocks climbed as commodity prices
    increased across the board.""",

    """The housing market shows signs of cooling as mortgage rates rise above 7%.
    Home sales declined for the third consecutive month while inventory levels
    gradually increase.""",

    """Cryptocurrency markets experienced significant volatility this week. Bitcoin
    and Ethereum prices fluctuated amid regulatory uncertainty and changing risk
    sentiment among investors.""",

    """Manufacturing data indicates growing economic slowdown concerns. Industrial
    production fell below expectations, raising fears about potential recession
    in major economies.""",

    """Bank earnings exceeded analyst expectations despite challenging market conditions.
    Financial institutions benefited from higher interest rates improving net interest
    margins.""",

    """Retail sales data showed consumer spending remains resilient despite inflation
    pressures. E-commerce continues to gain market share from traditional brick-and-mortar
    retailers.""",

    """Gold prices hit multi-month highs as investors seek safe haven assets. Precious
    metals attracted inflows amid market uncertainty and inflation concerns.""",

    """Pharmaceutical companies announced promising drug trial results. Healthcare sector
    gained on positive clinical data and regulatory approval expectations.""",
]


def sklearn_example():
    """Example using Scikit-learn LDA."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import numpy as np

    print("=" * 80)
    print("SCIKIT-LEARN LDA EXAMPLE")
    print("=" * 80)

    # 1. Vectorize documents
    vectorizer = CountVectorizer(
        max_features=100,
        min_df=1,
        max_df=0.95,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z]{3,}\b'  # Words with 3+ letters
    )

    doc_term_matrix = vectorizer.fit_transform(SAMPLE_DOCUMENTS)
    feature_names = vectorizer.get_feature_names_out()

    print(f"\n1. Vectorization:")
    print(f"   Documents: {len(SAMPLE_DOCUMENTS)}")
    print(f"   Vocabulary size: {len(feature_names)}")
    print(f"   Matrix shape: {doc_term_matrix.shape}")

    # 2. Train LDA
    n_topics = 5
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=100,
        learning_method='online',
        random_state=42,
        n_jobs=-1
    )

    print(f"\n2. Training LDA with {n_topics} topics...")
    lda.fit(doc_term_matrix)

    # 3. Display topics
    print(f"\n3. Extracted Topics:")
    print("-" * 80)
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-8:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"   Topic {topic_idx}: {', '.join(top_words)}")

    # 4. Document-topic distribution
    doc_topics = lda.transform(doc_term_matrix)
    print(f"\n4. Document-Topic Distribution (first 3 docs):")
    print("-" * 80)
    for doc_id in range(min(3, len(SAMPLE_DOCUMENTS))):
        dominant_topic = np.argmax(doc_topics[doc_id])
        probability = doc_topics[doc_id][dominant_topic]
        preview = SAMPLE_DOCUMENTS[doc_id][:60] + "..."
        print(f"   Doc {doc_id}: Topic {dominant_topic} ({probability:.2%})")
        print(f"   Preview: {preview}")
        print()

    # 5. Model metrics
    perplexity = lda.perplexity(doc_term_matrix)
    log_likelihood = lda.score(doc_term_matrix)
    print(f"5. Model Metrics:")
    print(f"   Perplexity: {perplexity:.2f}")
    print(f"   Log-likelihood: {log_likelihood:.2f}")


def gensim_example():
    """Example using Gensim LDA."""
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric, remove_stopwords, strip_short
    import numpy as np

    print("\n\n" + "=" * 80)
    print("GENSIM LDA EXAMPLE")
    print("=" * 80)

    # 1. Preprocess documents
    CUSTOM_FILTERS = [
        lambda x: x.lower(),
        strip_punctuation,
        strip_numeric,
        remove_stopwords,
        lambda x: strip_short(x, minsize=3)
    ]

    tokenized_docs = [preprocess_string(doc, CUSTOM_FILTERS)
                      for doc in SAMPLE_DOCUMENTS]

    print(f"\n1. Preprocessing:")
    print(f"   Documents: {len(tokenized_docs)}")
    print(f"   Sample tokens: {tokenized_docs[0][:8]}")

    # 2. Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=1, no_above=0.95, keep_n=None)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    print(f"\n2. Dictionary and Corpus:")
    print(f"   Dictionary size: {len(dictionary)}")
    print(f"   Corpus size: {len(corpus)} documents")

    # 3. Train LDA
    n_topics = 5
    print(f"\n3. Training LDA with {n_topics} topics...")

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=15,
        iterations=400,
        alpha='auto',
        eta='auto',
        random_state=42
    )

    # 4. Display topics
    print(f"\n4. Extracted Topics:")
    print("-" * 80)
    for topic_id in range(n_topics):
        words = lda.show_topic(topic_id, topn=8)
        words_str = ', '.join([f"{word}({prob:.3f})" for word, prob in words])
        print(f"   Topic {topic_id}: {words_str}")

    # 5. Document-topic distribution
    print(f"\n5. Document-Topic Distribution (first 3 docs):")
    print("-" * 80)
    for doc_id in range(min(3, len(corpus))):
        doc_topics = lda.get_document_topics(corpus[doc_id])
        if doc_topics:
            # Sort by probability
            doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
            dominant_topic, probability = doc_topics[0]
            preview = SAMPLE_DOCUMENTS[doc_id][:60] + "..."
            print(f"   Doc {doc_id}: Topic {dominant_topic} ({probability:.2%})")
            print(f"   Preview: {preview}")
            print()

    # 6. Model metrics
    perplexity = lda.log_perplexity(corpus)
    print(f"6. Model Metrics:")
    print(f"   Perplexity: {perplexity:.2f}")

    # 7. Compute coherence (optional)
    try:
        from gensim.models import CoherenceModel
        coherence_model = CoherenceModel(
            model=lda,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        print(f"   Coherence (C_v): {coherence:.4f}")
    except Exception as e:
        print(f"   Coherence calculation skipped: {e}")


if __name__ == "__main__":
    # Run both examples
    sklearn_example()
    gensim_example()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETED")
    print("=" * 80)
