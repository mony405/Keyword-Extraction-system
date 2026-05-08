import ast
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_text


def parse_gold(row):
    raw  = row['keyphrases']
    text = row['text']

    if pd.isna(raw):
        return set()

    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except:
            return set()

    keywords = set()

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and 'start' in item and 'end' in item:
                kw = text[item['start']:item['end']]
                keywords.add(kw.lower().strip())

    return keywords


def normalize_phrase(p):
    return preprocess_text(p)


def prf(predicted: list, gold: set) -> dict:
    pred = {normalize_phrase(p) for p in predicted}
    gold = {normalize_phrase(g) for g in gold}
    tp   = len(pred & gold)
    p    = tp / len(pred) if pred else 0
    r    = tp / len(gold) if gold else 0
    f1   = 2*p*r/(p+r) if p+r > 0 else 0
    return {'precision': p, 'recall': r, 'f1': f1}


def build_binary_labels(df_eval, method_col, gold_col, vectorizer):
    """
    For every token in the TF-IDF vocabulary, label it as:
      1 (keyword)     → if it appears in the method's extracted keywords
      0 (not keyword) → otherwise
    Do the same for gold labels, then return (y_true, y_pred) arrays.
    """
    vocab = list(vectorizer.get_feature_names_out())
    y_true_all, y_pred_all = [], []

    for _, row in df_eval.iterrows():
        gold_set  = row[gold_col]
        pred_list = row[method_col]
        if not isinstance(pred_list, list):
            pred_list = []
        pred_set = set(p.lower() for p in pred_list)

        for term in vocab:
            y_true_all.append(1 if term in gold_set else 0)
            y_pred_all.append(1 if term in pred_set else 0)

    return np.array(y_true_all), np.array(y_pred_all)