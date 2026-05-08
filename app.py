
import gradio as gr
import joblib
import pickle
import re

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


# ==========================================
# LOAD SAVED COMPONENTS
# ==========================================

print("Loading saved models...")

# TF-IDF Vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# KeyBERT model
kw_model = joblib.load("models/keybert_model.pkl")

# Stopwords + Lemmatizer
STOPWORDS = joblib.load("models/stopwords.pkl")
lemmatizer = joblib.load("models/lemmatizer.pkl")

# SBERT model name
with open("models/sbert_model_name.pkl", "rb") as f:
    sbert_name = pickle.load(f)

# Load SentenceTransformer
sbert_model = SentenceTransformer(sbert_name)

print("All components loaded successfully.")


# ==========================================
# PREPROCESSING
# ==========================================

def preprocess_text(text):
    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = text.split()

    cleaned_words = []

    for word in words:
        if word not in STOPWORDS:
            cleaned_words.append(lemmatizer.lemmatize(word))

    return " ".join(cleaned_words)


# ==========================================
# TF-IDF EXTRACTION
# ==========================================

def extract_tfidf_keywords(text, top_n=10):

    processed_text = preprocess_text(text)

    tfidf_matrix = vectorizer.transform([processed_text])

    scores = tfidf_matrix.toarray()[0]

    feature_names = vectorizer.get_feature_names_out()

    word_scores = list(zip(feature_names, scores))

    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)

    keywords = []

    for word, score in sorted_words:
        if score > 0:
            keywords.append((word, round(score, 4)))

    return keywords[:top_n]


# ==========================================
# EMBEDDING / KEYBERT EXTRACTION
# ==========================================

def extract_embedding_keywords(text, top_n=10):

    processed_text = preprocess_text(text)

    keywords = kw_model.extract_keywords(
        processed_text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n
    )

    return [(kw, round(score, 4)) for kw, score in keywords]


# ==========================================
# HIGHLIGHT KEYWORDS
# ==========================================

def highlight_keywords(text, keywords):

    highlighted_text = text

    for kw, _ in keywords:

        pattern = re.compile(re.escape(kw), re.IGNORECASE)

        highlighted_text = pattern.sub(
            f"""
            <span style="
                background:#8b5cf6;
                color:white;
                padding:2px 6px;
                border-radius:6px;
                font-weight:bold;
            ">
            {kw}
            </span>
            """,
            highlighted_text
        )

    return f"""
    <div style='
        padding:15px;
        line-height:2;
        font-size:18px;
        border-radius:10px;
    '>
    {highlighted_text}
    </div>
    """


# ==========================================
# FILE READING
# ==========================================

def read_uploaded_file(file):

    if file is None:
        return ""

    with open(file.name, "r", encoding="utf-8") as f:
        return f.read()


# ==========================================
# MAIN FUNCTION
# ==========================================

def generate_keyword_table(title, keywords, color):

    rows = ""

    for kw, score in keywords:
        rows += f"""
        <tr>
            <td style='padding:8px;'>{kw}</td>
            <td style='padding:8px; color:{color}; font-weight:bold;'>
                {score:.4f}
            </td>
        </tr>
        """

    return f"""
    <div style='width:48%; display:inline-block; vertical-align:top;'>

    <h3 style='color:{color};'>{title}</h3>

    <table style='width:100%; border-collapse:collapse;'>

    <tr>
        <th align='left'>Keyword</th>
        <th align='left'>Score</th>
    </tr>

    {rows}

    </table>
    </div>
    """


def keyword_extraction_pipeline(text, method):

    if not text.strip():
        return "", "", "Please enter text."

    if method == "TF-IDF":

        tfidf_keywords = extract_tfidf_keywords(text)

        output_html = generate_keyword_table(
            "TF-IDF Keywords",
            tfidf_keywords,
            "#8b5cf6"
        )

        highlighted = highlight_keywords(text, tfidf_keywords)

        return output_html, highlighted, "TF-IDF extraction completed."


    elif method == "Embedding-Based":

        embedding_keywords = extract_embedding_keywords(text)

        output_html = generate_keyword_table(
            "Embedding-Based Keywords",
            embedding_keywords,
            "#22c55e"
        )

        highlighted = highlight_keywords(text, embedding_keywords)

        return output_html, highlighted, "Embedding extraction completed."


    else:

        tfidf_keywords = extract_tfidf_keywords(text)

        embedding_keywords = extract_embedding_keywords(text)

        tfidf_html = generate_keyword_table(
            "TF-IDF Keywords",
            tfidf_keywords,
            "#8b5cf6"
        )

        embedding_html = generate_keyword_table(
            "Embedding-Based Keywords",
            embedding_keywords,
            "#22c55e"
        )

        combined_html = f"""
        <div style='display:flex; justify-content:space-between; gap:20px;'>
            {tfidf_html}
            {embedding_html}
        </div>
        """

        highlighted = highlight_keywords(
            text,
            tfidf_keywords + embedding_keywords
        )

        return combined_html, highlighted, "Comparison completed successfully."


# ==========================================
# LOAD FILE CONTENT
# ==========================================

def load_file_text(file):
    return read_uploaded_file(file)


# ==========================================
# GRADIO UI
# ==========================================

with gr.Blocks(theme=gr.themes.Soft(), title="Keyword Extraction System") as demo:

    gr.Markdown("""
<h1 style='text-align:center;'>Keyword Extraction System</h1>

<p style='text-align:center; font-size:18px;'>
Extract important keywords and keyphrases using TF-IDF,
Embedding-Based (SBERT + KeyBERT),
or compare both methods.
</p>
""")


    with gr.Row():

        with gr.Column(scale=1):

            text_input = gr.Textbox(
                label="Input Text",
                lines=12,
                placeholder="Enter your text here..."
            )

            # file_input = gr.File(
            #     label="Upload Text File (.txt)",
            #     file_types=[".txt"]
            # )

            method_dropdown = gr.Dropdown(
                choices=[
                    "TF-IDF",
                    "Embedding-Based",
                    "Compare Both"
                ],
                value="Compare Both",
                label="Extraction Method"
            )

            extract_button = gr.Button("Extract Keywords")


        with gr.Column(scale=1):

            keyword_output = gr.HTML(
    label="Extracted Keywords"
)


    highlighted_output = gr.HTML(
    label="Text with Highlighted Keywords"
)

    status_output = gr.Textbox(label="Status")


    # ======================================
    # EVENTS
    # ======================================

    # file_input.upload(
    #     fn=load_file_text,
    #     inputs=file_input,
    #     outputs=text_input
    # )


    extract_button.click(
        fn=keyword_extraction_pipeline,
        inputs=[text_input, method_dropdown],
        outputs=[
            keyword_output,
            highlighted_output,
            status_output
        ]
    )


# ==========================================
# LAUNCH
# ==========================================

if __name__ == "__main__":
    demo.launch()
