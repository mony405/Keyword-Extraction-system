from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vec():

    return TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True #log scaling,so frequent words won't dominate
    )

def sbert():
    print('Loading Sentence-BERT model...')
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sbert_model


     
