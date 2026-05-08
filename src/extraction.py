from src.preprocessing import preprocess_text



def extract_keywords_tfidf(text: str, vectorizer, top_n: int = 10):
    processed = preprocess_text(text)
    vec    = vectorizer.transform([processed])
    scores = vec.toarray()[0]
    top_idx = scores.argsort()[::-1][:top_n]
    return [(vectorizer.get_feature_names_out()[i], round(float(scores[i]), 4))
            for i in top_idx if scores[i] > 0]


def extract_keywords_keybert(text: str, model, top_n: int = 10, diversity: float = 0.5):
    """Extract keyphrases using KeyBERT with MMR diversity."""
    return model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        use_mmr=True,
        diversity=diversity,
        top_n=top_n
    )