
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS  = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str, lemmatize: bool = True) -> str:
    """
    Pipeline:
      1. Lowercase
      2. Remove punctuation / special characters
      3. Tokenize
      4. Remove stopwords & single-char tokens
      5. Lemmatize
    """
    text   = text.lower()
    text   = re.sub(r'[^a-z0-9\s]', ' ', text)
    text   = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)
