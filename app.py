import re
import string
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="MindWatch | Monitoring Isu Kesehatan Mental Gen Z",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — TEMA PINK AESTHETIC
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&family=Nunito:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
h1, h2, h3, h4 { font-family: 'Quicksand', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #1a0a14 0%, #2d1022 40%, #1f0d1a 100%);
    color: #f5e6f0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2a0f1f 0%, #1a0a14 100%);
    border-right: 1px solid rgba(255,105,180,0.2);
}
section[data-testid="stSidebar"] * { color: #f5c6e0 !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: #f5c6e0 !important; }

/* Hero */
.mw-hero {
    padding: 28px 32px;
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(220,80,150,0.3), rgba(255,160,200,0.12), rgba(180,60,120,0.2));
    border: 1px solid rgba(255,105,180,0.3);
    box-shadow: 0 10px 40px rgba(180,0,100,0.25);
    margin-bottom: 18px;
}
.mw-hero h1 {
    font-size: 28px; font-weight: 800; margin: 0 0 6px 0;
    background: linear-gradient(90deg, #ff69b4, #ffb6d9, #ff1493);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.mw-hero p { color: #f0c0d8; font-size: 14.5px; margin: 0; }

/* Glass card */
.mw-card {
    background: rgba(255,20,100,0.06);
    border: 1px solid rgba(255,105,180,0.15);
    backdrop-filter: blur(8px);
    border-radius: 18px;
    padding: 18px 20px;
    height: 100%;
    margin-bottom: 12px;
}
.mw-card h4 {
    margin-top: 0; font-size: 13px; color: #ff9ec8;
    font-weight: 700; letter-spacing: .4px; text-transform: uppercase;
}
.mw-card .big { font-size: 30px; font-weight: 800; color: #ffb6d9; }

/* Traffic lights */
.mw-light-wrap {
    display: flex; align-items: center; justify-content: center; gap: 22px;
    background: rgba(255,20,100,0.05);
    border: 1px solid rgba(255,105,180,0.15);
    border-radius: 20px; padding: 22px;
}
.mw-light { width: 46px; height: 46px; border-radius: 50%; opacity: 0.18; }
@keyframes pulse { 0%{transform:scale(1);} 50%{transform:scale(1.12);} 100%{transform:scale(1);} }

.mw-status-text { font-family: 'Quicksand', sans-serif; font-weight: 700; font-size: 20px; color: #ffb6d9; text-align: center; margin-top: 10px; }

/* Alert banner */
.mw-alert-banner {
    border-radius: 16px; padding: 16px 20px; margin: 14px 0;
    background: linear-gradient(90deg, rgba(239,68,68,0.15), rgba(239,68,68,0.04));
    border: 1px solid rgba(239,68,68,0.35);
    font-size: 14px; line-height: 1.6;
}
.mw-alert-banner b { color: #fca5a5; }

/* Metrics */
div[data-testid="stMetricValue"] { color: #ffb6d9 !important; }
div[data-testid="stMetric"] {
    background: rgba(255,20,100,0.06);
    border: 1px solid rgba(255,105,180,0.18);
    border-radius: 16px; padding: 10px 14px;
}
div[data-testid="stMetricLabel"] p { color: #f0c0d8 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,20,100,0.08);
    border-radius: 12px 12px 0 0;
    padding: 8px 16px; color: #f0c0d8;
    border: 1px solid rgba(255,105,180,0.15);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(220,80,150,0.4), rgba(255,160,200,0.2)) !important;
    color: #fff !important; border-color: rgba(255,105,180,0.4) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #c0507a, #e0709a);
    color: white; border: none; border-radius: 12px;
    font-family: 'Quicksand', sans-serif; font-weight: 700;
    padding: 8px 20px; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e0709a, #ff69b4);
    transform: translateY(-1px); box-shadow: 0 4px 20px rgba(220,80,150,0.4);
}

/* Selectbox & Multiselect */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: rgba(255,20,100,0.08) !important;
    border: 1px solid rgba(255,105,180,0.25) !important;
    border-radius: 10px !important; color: #f5e6f0 !important;
}

/* Text area */
.stTextArea > div > div > textarea {
    background: rgba(255,20,100,0.06) !important;
    border: 1px solid rgba(255,105,180,0.25) !important;
    border-radius: 12px !important; color: #f5e6f0 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,20,100,0.06);
    border: 1.5px dashed rgba(255,105,180,0.35);
    border-radius: 14px; padding: 12px;
}

hr { border-color: rgba(255,105,180,0.15) !important; }
[data-testid="stDataFrame"] { border: 1px solid rgba(255,105,180,0.18) !important; border-radius: 12px !important; }

.mw-footer {
    margin-top: 30px; padding: 18px; border-radius: 16px;
    background: rgba(255,20,100,0.06); border: 1px solid rgba(255,105,180,0.15);
    font-size: 12.5px; color: #d09ab8; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ====================  KAMUS DOSEN (ASLI)  ===================
# ============================================================
# --- Stopwords versi notebook LDA (UAS) ---
STOPWORDS_UAS = [
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'adalah', 'ada', 'juga',
    'saya', 'aku', 'gue', 'gw', 'kamu', 'kau', 'lo', 'lu', 'tapi', 'akan', 'sudah',
    'untuk', 'karena', 'tidak', 'ya', 'deh', 'dong', 'lah', 'kan', 'kak', 'sama', 'buat',
    'mau', 'gimana', 'atau', 'terus', 'habis', 'pas', 'nih', 'aja', 'ok', 'iya', 'banget',
    'bgt', 'amp', 'kalo', 'kalau', 'wkwkwk', 'jadi', 'loh', 'masih', 'shi', 'yaa', 'lagi',
    'tuh', 'gak', 'nggak', 'haha', 'hihi', 'hehe', 'woy', 'gini', 'trs', 'kayak', 'kek',
    'kok', 'apa', 'depresi', 'cemas', 'kecemasan', 'stres', 'bisa', 'bikin', 'kalau', 'orang'
]

# --- Stopwords versi notebook K-Means (NLTK + custom), dengan fallback offline ---
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    try:
        STOP_WORDS_ID = set(nltk_stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        STOP_WORDS_ID = set(nltk_stopwords.words('indonesian'))
except Exception:
    # Fallback offline jika NLTK corpus tidak bisa diunduh (mis. tidak ada akses internet di server)
    STOP_WORDS_ID = {
        'yang','dan','di','ke','dari','ini','itu','dengan','adalah','ada','juga','saya','aku',
        'kamu','dia','mereka','kami','kita','untuk','pada','akan','sudah','belum','tidak','bukan',
        'atau','karena','jika','kalau','agar','supaya','namun','tetapi','tapi','serta','maka',
        'oleh','dalam','luar','atas','bawah','antara','setelah','sebelum','sejak','hingga','sampai',
        'sebagai','seperti','bagai','adapun','meski','meskipun','walaupun','sehingga','yaitu','yakni'
    }

CUSTOM_STOPWORDS = {
    'yg', 'ya', 'kayak', 'dgn', 'udah', 'itu', 'ini', 'ke', 'di', 'kita', 'bgt', 'tuh',
    'gak', 'nggak', 'nya', 'sih', 'kok', 'apa', 'siapa', 'atau', 'yang', 'dengan',
    'kamu', 'dia', 'aku', 'krn', 'sm', 'tp', 'jg', 'karena', 'jadi', 'untuk', 'pada',
    'gw', 'gue', 'lu', 'banget', 'kalo', 'deh', 'aja', 'dr', 'ada', 'dari', 'depresi', 'cemas'
}
ALL_STOPWORDS_KMEANS = STOP_WORDS_ID.union(CUSTOM_STOPWORDS)

# --- Kata kunci sentimen (notebook eksplorasi awal) ---
POSITIVE_WORDS = [
    'good', 'better', 'happy', 'happier', 'calm', 'peace',
    'peaceful', 'relieved', 'relief', 'hope', 'hopeful',
    'healing', 'healed', 'recover', 'recovered', 'recovery',
    'strong', 'stronger', 'support', 'supported', 'help',
    'helped', 'helpful', 'love', 'loved', 'grateful',
    'thankful', 'fine', 'okay', 'ok', 'smile', 'positive',
    'safe', 'comfort', 'comfortable', 'proud', 'improving'
]

NEGATIVE_WORDS = [
    'depression', 'depressed', 'depressing', 'anxiety', 'anxious',
    'burnout', 'burned out', 'stress', 'stressed', 'stressful',
    'sad', 'sadness', 'cry', 'crying', 'tears', 'lonely',
    'alone', 'tired', 'exhausted', 'drained', 'empty',
    'hopeless', 'helpless', 'worthless', 'overthinking',
    'panic', 'panicking', 'scared', 'afraid', 'fear',
    'hurt', 'pain', 'broken', 'trauma', 'traumatized',
    'suffering', 'struggle', 'struggling', 'hard', 'heavy',
    'mental', 'mental health', 'not okay', 'not fine',
    'give up', 'giving up', 'tired of life', 'want to disappear'
]

# --- Kata kunci urgensi (notebook eksplorasi awal) ---
URGENT_WORDS = [
    'give up', 'giving up', 'want to give up',
    'not strong enough', "can't do this", 'cannot do this',
    'tired of life', 'want to disappear',
    'need help', 'please help', 'help me',
    'hopeless', 'helpless', 'worthless',
    'not okay', 'not fine', 'panic attack',
    'severe depression', 'depressed so bad',
    'mental breakdown', 'breaking down'
]

# --- Kamus trending topic untuk LDA (vocabulary tetap) ---
KAMUS_TRENDING_TOPIC = [
    'orangtua', 'keluarga', 'ibu', 'ayah', 'rumah', 'cerai', 'berantem', 'kakak', 'adik', 'ortu',
    'kuliah', 'skripsi', 'tugas', 'akademik', 'nilai', 'dosen', 'ujian', 'sekolah', 'lulus', 'semester',
    'uang', 'finansial', 'kerja', 'gaji', 'biaya', 'miskin', 'hutang', 'ekonomi', 'tabungan', 'dana'
]

LABEL_TREN_MAPPING = {
    0: "Trending: Tekanan Akademik & Perkuliahan",
    1: "Trending: Krisis Finansial & Ekonomi",
    2: "Trending: Konflik Keluarga & Rumah"
}

# --- Kamus akar masalah untuk K-Means (vocabulary tetap) ---
KAMUS_AKAR_MASALAH = [
    'orangtua', 'keluarga', 'ibu', 'ayah', 'rumah', 'cerai', 'berantem', 'kakak', 'adik',  # Keluarga
    'kuliah', 'skripsi', 'tugas', 'akademik', 'nilai', 'dosen', 'ujian', 'sekolah', 'lulus',  # Akademik
    'uang', 'finansial', 'kerja', 'gaji', 'biaya', 'miskin', 'hutang', 'ekonomi', 'tabungan'  # Finansial
]

KMEANS_LABEL_MAPPING = {
    0: "Klaster Masalah Keluarga",
    1: "Klaster Tekanan Akademik",
    2: "Klaster Finansial"
}

# --- Frasa support (untuk SNA "respon positif") ---
SUPPORT_PHRASES = {
    "semangat", "kamu kuat", "gpp", "gapapa", "tidak apa apa", "ada aku", "dm aja", "cerita yuk",
    "pelukan", "peluk", "stay strong", "kamu berharga", "tetap semangat", "jangan menyerah",
    "kita disini", "sini cerita", "semangat ya", "kamu hebat", "peluk jauh", "kamu tidak sendirian",
    "boleh cerita", "aku dengerin", "semoga membaik", "semoga lekas membaik", "tetap kuat",
    "kamu pasti bisa", "jangan lupa makan", "jangan lupa istirahat", "sehat selalu",
}

CRISIS_RESOURCES = """
📞 **SEJIWA (Kemenkes):** 119 ext 8  |  📞 **Into The Light:** intothelightid.org  |  📞 **Yayasan Pulih:** (021) 78842580
"""

PINK_COLORS = ["#ff69b4", "#ff1493", "#ffb6d9", "#c0507a", "#e0709a", "#ff9ec8"]
COLOR_MAP_URGENCY = {
    "Need Immediate Help": "#ef4444",
    "Light Venting": "#ff69b4",
}
COLOR_MAP_SENTIMENT = {
    "Positive": "#10b981",
    "Negative": "#ef4444",
    "Neutral": "#f59e0b",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f5e6f0",
    font_family="Nunito",
)

# ============================================================
# ====================  FUNGSI PRAPROSES (ASLI DOSEN)  =========
# ============================================================

def clean_text_basic(full_text):
    """Versi notebook eksplorasi (sentimen & urgensi versi awal — Bahasa Inggris)."""
    full_text = str(full_text).lower()
    full_text = re.sub(r"http\S+", "", full_text)
    full_text = re.sub(r"@\w+", "", full_text)
    full_text = re.sub(r"[^a-zA-Z\s]", "", full_text)
    return full_text


def bersihkan_teks_lda(text):
    """Persis fungsi notebook LDA dosen."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)

    tokens = text.split()
    clean_tokens = [w for w in tokens if w not in STOPWORDS_UAS and len(w) > 2]
    return " ".join(clean_tokens)


def clean_text_proper(text):
    """Persis fungsi notebook K-Means dosen."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    filtered_words = [w for w in words if w not in ALL_STOPWORDS_KMEANS and len(w) > 2]
    return " ".join(filtered_words)


def sentiment_score(text):
    score = 0
    text = str(text).lower()
    for word in POSITIVE_WORDS:
        if word in text:
            score += 1
    for word in NEGATIVE_WORDS:
        if word in text:
            score -= 1
    return score


def label_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'


def classify_urgency(text):
    text = str(text).lower()
    for word in URGENT_WORDS:
        if word in text:
            return 'Need Immediate Help'
    return 'Light Venting'


def detect_support(text_lower):
    return any(p in text_lower for p in SUPPORT_PHRASES)


# ============================================================
# AUTO-DETEKSI KOLOM
# ============================================================
def auto_detect(df, candidates):
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return None


def detect_text_col(df):
    return auto_detect(df, ["full_text", "text", "tweet", "content", "caption", "isi", "teks", "curhatan"])


def detect_username_col(df):
    return auto_detect(df, ["username", "user", "account", "akun", "screen_name", "author", "nama_user"])


def detect_parent_col(df):
    return auto_detect(df, ["in_reply_to_screen_name", "in_reply_to_user", "in_reply_to", "reply_to", "parent", "membalas"])


def detect_timestamp_col(df):
    return auto_detect(df, ["created_at", "timestamp", "date", "tanggal", "waktu", "time"])


def detect_userid_col(df):
    return auto_detect(df, ["user_id_str", "user_id", "id_user", "userid"])


# ============================================================
# LDA — PERSIS PIPELINE NOTEBOOK DOSEN
# ============================================================
@st.cache_data(show_spinner=False)
def run_lda_dosen(clean_texts):
    count_vectorizer = CountVectorizer(vocabulary=list(set(KAMUS_TRENDING_TOPIC)))
    X_bow = count_vectorizer.fit_transform(clean_texts)
    fitur_kata = count_vectorizer.get_feature_names_out()

    lda_model = LatentDirichletAllocation(
        n_components=3,
        random_state=42,
        learning_method='online',
        max_iter=30
    )
    doc_topic_distribution = lda_model.fit_transform(X_bow)

    log_lik = lda_model.score(X_bow)
    perplexity = lda_model.perplexity(X_bow)

    topic_top_words = {}
    for idx, komponen in enumerate(lda_model.components_):
        top_idx = komponen.argsort()[::-1][:8]
        topic_top_words[idx] = [fitur_kata[i] for i in top_idx]

    dominant_topics = doc_topic_distribution.argmax(axis=1)
    return dominant_topics, topic_top_words, log_lik, perplexity


# ============================================================
# K-MEANS — PERSIS PIPELINE NOTEBOOK DOSEN
# ============================================================
@st.cache_data(show_spinner=False)
def run_kmeans_dosen(clean_texts):
    vectorizer = TfidfVectorizer(max_features=1000, vocabulary=list(set(KAMUS_AKAR_MASALAH)))
    X_tfidf = vectorizer.fit_transform(clean_texts)

    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)

    optimal_k = 3
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(X_2d)

    sil = silhouette_score(X_2d, labels) if len(set(labels)) > 1 else 0
    dbi = davies_bouldin_score(X_2d, labels) if len(set(labels)) > 1 else 0

    return labels, X_2d, kmeans_final, sil, dbi


# ============================================================
# ML KLASIFIKASI (TF-IDF + Naive Bayes + SVM) — PERSIS NOTEBOOK
# ============================================================
def run_ml_classification(X_text, y_label):
    """Mengembalikan dict hasil akurasi per model, atau None jika kelas < 2."""
    if y_label.nunique() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_label, test_size=0.2, random_state=42, stratify=y_label
    )
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    models = {'Naive Bayes': MultinomialNB(), 'SVM Linear': LinearSVC()}
    results = {}
    for nama_model, model in models.items():
        model.fit(X_train_tfidf, y_train)
        prediksi = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, prediksi)
        report = classification_report(y_test, prediksi, output_dict=True, zero_division=0)
        results[nama_model] = {"accuracy": acc, "report": report}
    return results


# ============================================================
# SNA — PERSIS POLA NOTEBOOK DOSEN (edge list: username -> mention)
# ============================================================
def build_network_dosen(df, text_col, source_col):
    """
    Mengikuti pola notebook dosen persis:
        for i, row in df.iterrows():
            mentions = re.findall(r"@(\\w+)", row["full_text"])
            for mention in mentions:
                edges.append((row["username"], mention))
    Bedanya: source_col bisa fallback ke user_id_str jika username kosong semua,
    dan kita tandai edge yang mengandung kata dukungan agar bisa dianalisis support system.
    """
    edges = []
    edge_support_flag = []

    for i, row in df.iterrows():
        mentions = re.findall(r"@(\w+)", str(row[text_col]))
        source = row[source_col]
        is_support = detect_support(str(row[text_col]).lower())
        for mention in mentions:
            if pd.isna(source):
                continue
            source_str = str(source)
            if mention == source_str:
                continue
            edges.append((source_str, mention))
            edge_support_flag.append(is_support)

    G = nx.DiGraph()
    for (src, tgt), is_support in zip(edges, edge_support_flag):
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += 1
            if is_support:
                G[src][tgt]["support"] = True
        else:
            G.add_edge(src, tgt, weight=1, support=is_support)

    return G, edges


def top_support_accounts(G, top_n=10):
    rows = [u for u, v, d in G.edges(data=True) if d.get("support")]
    if not rows:
        return pd.DataFrame(columns=["Akun", "Jumlah Respon Positif"])
    counts = Counter(rows)
    return pd.DataFrame(counts.most_common(top_n), columns=["Akun", "Jumlah Respon Positif"])


def most_supported_accounts(G, top_n=10):
    rows = [v for u, v, d in G.edges(data=True) if d.get("support")]
    if not rows:
        return pd.DataFrame(columns=["Akun", "Jumlah Dukungan Diterima"])
    counts = Counter(rows)
    return pd.DataFrame(counts.most_common(top_n), columns=["Akun", "Jumlah Dukungan Diterima"])


def plot_network(G, max_nodes=60):
    if G.number_of_nodes() == 0:
        return None
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top_nodes]).copy()

    pos = nx.spring_layout(G, k=0.5, seed=42)
    edge_x, edge_y, sup_x, sup_y = [], [], [], []

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        if d.get("support"):
            sup_x += [x0, x1, None]; sup_y += [y0, y1, None]
        else:
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.6, color="rgba(255,105,180,0.2)"), hoverinfo="none")
    sup_trace = go.Scatter(x=sup_x, y=sup_y, mode="lines",
                            line=dict(width=1.6, color="rgba(255,182,217,0.6)"), hoverinfo="none")

    degs = dict(G.degree())
    max_deg = max(degs.values()) if degs else 1
    node_x, node_y, node_txt, node_sz, node_col = [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        deg = degs[n]
        node_txt.append(f"@{n}<br>Koneksi: {deg}")
        node_sz.append(10 + 30 * (deg / max_deg))
        node_col.append(deg)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_txt,
        marker=dict(size=node_sz, color=node_col, colorscale="RdPu", reversescale=True,
                    line=dict(width=1, color="rgba(255,255,255,0.3)"), showscale=False)
    )

    fig = go.Figure(data=[edge_trace, sup_trace, node_trace])
    fig.update_layout(
        showlegend=False, hovermode="closest",
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )
    return fig


# ============================================================
# UI — HERO
# ============================================================
st.markdown("""
<div class="mw-hero">
  <h1>🧠 MindWatch — Monitoring Isu Kesehatan Mental Gen Z</h1>
  <p>Sentimen & Urgensi (Lexicon + ML) · LDA Trending Topic · K-Means Akar Masalah · Social Network Analysis (SNA)</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🌸 Upload Dataset")
    uploaded_file = st.file_uploader("Upload file CSV (format data_depresi.csv)", type=["csv"])
    st.caption("Tidak upload file? Dashboard otomatis memakai `data_depresi.csv` bawaan project.")
    st.divider()
    st.markdown("### ℹ️ Tentang")
    st.caption(
        "**Tema 7** — Monitoring Isu Kesehatan Mental Gen Z\n\n"
        "Komponen sesuai materi:\n"
        "- Sentimen & Klasifikasi Urgensi (lexicon + TF-IDF & Naive Bayes/SVM)\n"
        "- LDA Trending Topic (vocabulary akademik/finansial/keluarga)\n"
        "- K-Means Klaster Akar Masalah (TF-IDF + SVD)\n"
        "- SNA Support System Network (edge list mention)"
    )
    st.divider()
    st.markdown("### 🆘 Krisis? Hubungi:")
    st.caption(CRISIS_RESOURCES)

# ============================================================
# LOAD DATA
# ============================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("data_depresi.csv")
        st.info("📂 Memakai dataset bawaan **data_depresi.csv**. Upload file lain di sidebar untuk mengganti.")
    except FileNotFoundError:
        df = None

if df is not None:
    st.success(f"✅ Dataset dimuat — **{len(df):,} baris**, **{len(df.columns)} kolom**")

    # ---- Konfigurasi kolom ----
    with st.expander("⚙️ Konfigurasi Kolom Dataset", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            default_text = detect_text_col(df)
            text_col = st.selectbox("Kolom Teks:", df.columns,
                                     index=list(df.columns).index(default_text) if default_text else 0)
            default_user = detect_username_col(df)
            username_col = st.selectbox("Kolom Username:", ["(tidak ada)"] + list(df.columns),
                                         index=(list(df.columns).index(default_user) + 1) if default_user else 0)
            username_col = None if username_col == "(tidak ada)" else username_col
        with c2:
            default_uid = detect_userid_col(df)
            userid_col = st.selectbox("Kolom User ID (fallback jika username kosong):",
                                       ["(tidak ada)"] + list(df.columns),
                                       index=(list(df.columns).index(default_uid) + 1) if default_uid else 0)
            userid_col = None if userid_col == "(tidak ada)" else userid_col
            default_ts = detect_timestamp_col(df)
            ts_col = st.selectbox("Kolom Waktu (opsional):", ["(tidak ada)"] + list(df.columns),
                                   index=(list(df.columns).index(default_ts) + 1) if default_ts else 0)
            ts_col = None if ts_col == "(tidak ada)" else ts_col

        st.caption(
            "💡 Pada `data_depresi.csv` bawaan, kolom **username** kosong di seluruh baris. "
            "Dashboard otomatis memakai **user_id_str** sebagai identitas akun pengganti untuk SNA."
        )

    # Tentukan kolom sumber SNA: username kalau terisi, kalau tidak fallback ke user id
    if username_col and df[username_col].notna().sum() > 0:
        sna_source_col = username_col
    elif userid_col:
        sna_source_col = userid_col
    else:
        sna_source_col = None

    # ---- Preprocessing & analisis (mengikuti notebook dosen apa adanya) ----
    with st.spinner("🌸 Memproses data sesuai pipeline notebook..."):
        # --- versi clean_text dasar (untuk sentimen & urgensi versi lexicon EN) ---
        df["clean_text"] = df[text_col].astype(str).apply(clean_text_basic)

        # --- sentimen ---
        df["sentiment_score"] = df["clean_text"].apply(sentiment_score)
        df["sentiment"] = df["sentiment_score"].apply(label_sentiment)

        # --- urgensi (lexicon) ---
        df["urgency_classification"] = df["clean_text"].apply(classify_urgency)

        # --- versi preprocessing LDA (notebook UAS) ---
        df["text_clean_lda"] = df[text_col].astype(str).apply(bersihkan_teks_lda)
        df_lda = df[df["text_clean_lda"].str.strip() != ""].reset_index(drop=True)

        # --- versi preprocessing K-Means (notebook UAS) ---
        df["text_clean_kmeans"] = df[text_col].astype(str).apply(clean_text_proper)

        if ts_col:
            df["_ts_parsed"] = pd.to_datetime(df[ts_col], errors="coerce")

        # --- LDA ---
        dom_topics, topic_words, log_lik, perplexity = run_lda_dosen(df_lda["text_clean_lda"].tolist())
        df_lda["topik_dominan"] = dom_topics
        df_lda["label_trending_topic"] = df_lda["topik_dominan"].map(LABEL_TREN_MAPPING)

        # --- K-Means ---
        km_labels, X_2d, km_final, sil, dbi = run_kmeans_dosen(df["text_clean_kmeans"].tolist())
        df["kmeans_cluster"] = km_labels
        df["akar_masalah_label"] = df["kmeans_cluster"].map(KMEANS_LABEL_MAPPING)

        # --- SNA ---
        if sna_source_col:
            G, edges = build_network_dosen(df, text_col, sna_source_col)
        else:
            G, edges = nx.DiGraph(), []

    total = len(df)
    n_urgent = (df["urgency_classification"] == "Need Immediate Help").sum()
    n_light = (df["urgency_classification"] == "Light Venting").sum()
    pct_urgent = n_urgent / total * 100 if total else 0

    if pct_urgent >= 15:
        light_class, status_text = "on-red", "⚠️ SIAGA TINGGI"
    elif pct_urgent >= 5:
        light_class, status_text = "on-yellow", "👁️ PERLU DIPANTAU"
    else:
        light_class, status_text = "on-green", "✅ TERKENDALI"

    light_styles = {
        "red": "background:#ef4444; opacity:{op}; {glow}",
        "yellow": "background:#f59e0b; opacity:{op}; {glow}",
        "green": "background:#10b981; opacity:{op}; {glow}",
    }
    glow = {
        "red": "box-shadow:0 0 35px 8px rgba(239,68,68,0.75);animation:pulse 1.4s infinite;",
        "yellow": "box-shadow:0 0 35px 8px rgba(245,158,11,0.6);",
        "green": "box-shadow:0 0 30px 6px rgba(16,185,129,0.55);",
    }

    # ---- Indikator status ----
    st.markdown("### 🚦 Indikator Urgensi")
    col_light, col_kpi = st.columns([1, 2.5])

    with col_light:
        active = light_class.replace("on-", "")
        st.markdown(f"""
        <div class="mw-light-wrap">
          <div class="mw-light" style="{light_styles['red'].format(op=1 if active=='red' else 0.15, glow=glow['red'] if active=='red' else '')}"></div>
          <div class="mw-light" style="{light_styles['yellow'].format(op=1 if active=='yellow' else 0.15, glow=glow['yellow'] if active=='yellow' else '')}"></div>
          <div class="mw-light" style="{light_styles['green'].format(op=1 if active=='green' else 0.15, glow=glow['green'] if active=='green' else '')}"></div>
        </div>
        <p class="mw-status-text">{status_text}</p>
        """, unsafe_allow_html=True)

    with col_kpi:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Tweet", f"{total:,}")
        k2.metric("🚨 Need Immediate Help", int(n_urgent), f"{pct_urgent:.1f}%")
        k3.metric("💬 Light Venting", int(n_light))

    if n_urgent > 0:
        st.markdown(f"""
        <div class="mw-alert-banner">
        <b>⚠️ Peringatan:</b> Terdeteksi <b>{int(n_urgent)} curhatan berisiko tinggi (Need Immediate Help)</b>.
        Segmen ini memerlukan perhatian segera.<br><small>{CRISIS_RESOURCES}</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Sentimen & Urgensi",
        "🔬 LDA Trending Topic",
        "🧩 K-Means Akar Masalah",
        "🕸️ Social Network (SNA)",
        "📋 Data & Export",
        "🤖 ML Classifier",
    ])

    # -------- TAB 1: SENTIMEN & URGENSI --------
    with tab1:
        st.subheader("Distribusi Sentimen")
        c1, c2 = st.columns(2)
        with c1:
            sent_count = df["sentiment"].value_counts().reset_index()
            sent_count.columns = ["Sentimen", "Jumlah"]
            fig_pie = px.pie(sent_count, names="Sentimen", values="Jumlah", hole=0.55,
                              color="Sentimen", color_discrete_map=COLOR_MAP_SENTIMENT)
            fig_pie.update_layout(**PLOTLY_LAYOUT, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_bar = px.bar(sent_count, x="Sentimen", y="Jumlah", color="Sentimen",
                              color_discrete_map=COLOR_MAP_SENTIMENT, text="Jumlah")
            fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.caption(
            "ℹ️ Sentimen dihitung dengan **lexicon-based scoring** (daftar kata positif vs negatif Bahasa Inggris) "
            "sesuai notebook eksplorasi awal — cocok untuk caption/tweet campuran ID-EN."
        )

        st.subheader("Klasifikasi Tingkat Urgensi")
        c3, c4 = st.columns(2)
        with c3:
            urg_count = df["urgency_classification"].value_counts().reset_index()
            urg_count.columns = ["Urgensi", "Jumlah"]
            fig_upie = px.pie(urg_count, names="Urgensi", values="Jumlah", hole=0.55,
                               color="Urgensi", color_discrete_map=COLOR_MAP_URGENCY)
            fig_upie.update_layout(**PLOTLY_LAYOUT, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_upie, use_container_width=True)
        with c4:
            cross = df.groupby(["sentiment", "urgency_classification"]).size().reset_index(name="Jumlah")
            fig_cross = px.bar(cross, x="sentiment", y="Jumlah", color="urgency_classification",
                                color_discrete_map=COLOR_MAP_URGENCY, barmode="group",
                                labels={"sentiment": "Sentimen", "urgency_classification": "Urgensi"})
            fig_cross.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_cross, use_container_width=True)

        if ts_col and "_ts_parsed" in df.columns and df["_ts_parsed"].notna().sum() > 0:
            st.subheader("Tren Urgensi dari Waktu ke Waktu")
            trend_df = df.dropna(subset=["_ts_parsed"]).copy()
            trend_df["periode"] = trend_df["_ts_parsed"].dt.to_period("D").astype(str)
            trend_pivot = trend_df.groupby(["periode", "urgency_classification"]).size().unstack(fill_value=0).reset_index()
            fig_trend = go.Figure()
            for label, color in COLOR_MAP_URGENCY.items():
                if label in trend_pivot.columns:
                    fig_trend.add_trace(go.Scatter(
                        x=trend_pivot["periode"], y=trend_pivot[label],
                        mode="lines+markers", name=label, line=dict(color=color, width=2.5)
                    ))
            fig_trend.update_layout(**PLOTLY_LAYOUT, xaxis_title="Tanggal", yaxis_title="Jumlah")
            st.plotly_chart(fig_trend, use_container_width=True)

    # -------- TAB 2: LDA --------
    with tab2:
        st.subheader("🔬 Latent Dirichlet Allocation (LDA) — Trending Topic")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="mw-card">
            <h4>Metrik Evaluasi Model LDA</h4>
            <p style="color:#f0c0d8; font-size:14px;">
            📊 <b>Log-Likelihood:</b> {log_lik:.2f}<br>
            📉 <b>Perplexity:</b> {perplexity:.2f} <small>(lebih kecil = lebih baik)</small><br>
            🔢 <b>Jumlah Topik:</b> 3 (sesuai tema: Akademik, Finansial, Keluarga)
            </p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="mw-card">
            <h4>Pipeline</h4>
            <p style="color:#f0c0d8; font-size:13px;">
            1️⃣ Preprocessing teks (case folding, hapus URL/mention/hashtag/angka/tanda baca)<br>
            2️⃣ <b>CountVectorizer</b> dengan <b>vocabulary tetap</b> (kamus trending topic)<br>
            3️⃣ <b>LatentDirichletAllocation</b> (n_components=3, learning_method='online')
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Kata Kunci Tiap Topik")
        cols_topics = st.columns(3)
        for i, col in enumerate(cols_topics):
            with col:
                words_str = " · ".join(topic_words[i])
                st.markdown(f"""
                <div class="mw-card">
                <h4>{LABEL_TREN_MAPPING[i]}</h4>
                <p style="color:#ffb6d9; font-size:13px; line-height:1.8;">{words_str}</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("Distribusi Dokumen per Trending Topic")
        topic_dist = df_lda["label_trending_topic"].value_counts().reset_index()
        topic_dist.columns = ["Trending Topic", "Jumlah"]
        fig_lda = px.bar(topic_dist, x="Trending Topic", y="Jumlah",
                          color="Jumlah", color_continuous_scale=[[0, "#2d1022"], [1, "#ff69b4"]],
                          text="Jumlah")
        fig_lda.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, xaxis_title="", yaxis_title="Jumlah Dokumen")
        st.plotly_chart(fig_lda, use_container_width=True)

        st.caption(
            f"ℹ️ {len(df) - len(df_lda)} baris dikecualikan dari analisis LDA karena teks kosong setelah preprocessing."
        )

    # -------- TAB 3: K-MEANS --------
    with tab3:
        st.subheader("🧩 K-Means Clustering — Akar Masalah Penyebab Kecemasan")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="mw-card">
            <h4>Metrik Evaluasi K-Means (K=3)</h4>
            <p style="color:#f0c0d8; font-size:14px;">
            🔵 <b>Silhouette Score:</b> {sil:.4f} <small>(mendekati 1 = baik)</small><br>
            📉 <b>Davies-Bouldin Index:</b> {dbi:.4f} <small>(&lt;1.0 = ideal)</small>
            </p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="mw-card">
            <h4>Pipeline</h4>
            <p style="color:#f0c0d8; font-size:13px;">
            1️⃣ <b>TF-IDF Vectorizer</b> dengan <b>vocabulary tetap</b> (kamus akar masalah)<br>
            2️⃣ <b>TruncatedSVD</b> → reduksi dimensi ke 2D<br>
            3️⃣ <b>K-Means</b> (K=3: Keluarga, Akademik, Finansial)
            </p>
            </div>
            """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Visualisasi Klaster (2D)")
            df_plot = pd.DataFrame({"x": X_2d[:, 0], "y": X_2d[:, 1], "Klaster": df["akar_masalah_label"]})
            centroids = km_final.cluster_centers_
            fig_scatter = px.scatter(df_plot, x="x", y="y", color="Klaster",
                                      color_discrete_sequence=PINK_COLORS,
                                      labels={"x": "SVD Komponen 1", "y": "SVD Komponen 2"})
            fig_scatter.add_trace(go.Scatter(
                x=centroids[:, 0], y=centroids[:, 1], mode="markers",
                marker=dict(symbol="x", size=16, color="white", line=dict(width=2, color="white")),
                name="Centroid",
            ))
            fig_scatter.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c2:
            st.subheader("Distribusi Jumlah Tweet per Klaster")
            klaster_count = df["akar_masalah_label"].value_counts().reset_index()
            klaster_count.columns = ["Klaster", "Jumlah"]
            fig_kb = px.bar(klaster_count.sort_values("Jumlah"), x="Jumlah", y="Klaster",
                             orientation="h", color="Jumlah",
                             color_continuous_scale=[[0, "#2d1022"], [1, "#ff69b4"]])
            fig_kb.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_kb, use_container_width=True)

        st.subheader("Kata Dominan per Klaster (Top-8 Words)")
        cols_km = st.columns(3)
        for cluster_id in range(3):
            with cols_km[cluster_id]:
                cluster_texts = df[df["kmeans_cluster"] == cluster_id]["text_clean_kmeans"]
                all_words = " ".join(cluster_texts).split()
                word_counts = Counter(all_words)
                top_words = [w for w, _ in word_counts.most_common(8)]
                n_docs = len(cluster_texts)
                words_str = " · ".join(top_words) if top_words else "—"
                st.markdown(f"""
                <div class="mw-card">
                <h4>{KMEANS_LABEL_MAPPING[cluster_id]}</h4>
                <p style="color:#e0c0d0; font-size:12px;">📄 {n_docs} dokumen</p>
                <p style="color:#ffb6d9; font-size:13px; line-height:1.8;">{words_str}</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("WordCloud per Klaster")
        cluster_pick = st.selectbox("Pilih Klaster:", [KMEANS_LABEL_MAPPING[i] for i in range(3)])
        cluster_id_pick = [k for k, v in KMEANS_LABEL_MAPPING.items() if v == cluster_pick][0]
        text_blob = " ".join(df[df["kmeans_cluster"] == cluster_id_pick]["text_clean_kmeans"])
        if text_blob.strip():
            wc = WordCloud(width=1100, height=380, background_color=None, mode="RGBA",
                            colormap="RdPu", max_words=80).generate(text_blob)
            fig_wc, ax_wc = plt.subplots(figsize=(12, 4))
            fig_wc.patch.set_alpha(0)
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("Tidak cukup data untuk WordCloud pada klaster ini.")

    # -------- TAB 4: SNA --------
    with tab4:
        st.subheader("🕸️ Social Network Analysis — Support System Network")
        st.caption("Edge list dibangun dari pola `(username, mention)` per baris tweet — persis pola notebook. "
                    "Warna edge **pink** = interaksi biasa · **pink terang** = mengandung kata dukungan/semangat.")

        if not sna_source_col:
            st.warning("Tidak ada kolom username/user_id yang bisa dipakai sebagai identitas akun untuk SNA.")
        elif G.number_of_nodes() == 0:
            st.warning("Tidak ada relasi (@mention) terdeteksi pada teks dataset ini.")
        else:
            st.caption(f"Sumber identitas akun yang dipakai: **{sna_source_col}** "
                       f"({'username asli' if sna_source_col == username_col else 'fallback ke user ID karena username kosong'})")

            colA, colB = st.columns([1.6, 1])
            with colA:
                fig_net = plot_network(G)
                st.plotly_chart(fig_net, use_container_width=True)
            with colB:
                st.markdown("**🏅 Top Pemberi Dukungan**")
                st.dataframe(top_support_accounts(G), use_container_width=True, hide_index=True)
                st.markdown("**🙋 Top Penerima Dukungan**")
                st.dataframe(most_supported_accounts(G), use_container_width=True, hide_index=True)

            g1, g2, g3 = st.columns(3)
            g1.metric("Node (Akun)", G.number_of_nodes())
            g2.metric("Edge (Relasi Mention)", G.number_of_edges())
            n_sup_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("support"))
            g3.metric("Edge Supportif", n_sup_edges)

            st.subheader("📊 Centrality Metrics")
            deg_cen = nx.degree_centrality(G)
            btw_cen = nx.betweenness_centrality(G)
            try:
                eig_cen = nx.eigenvector_centrality(G, max_iter=500)
            except Exception:
                eig_cen = {}

            top5_deg = sorted(deg_cen.items(), key=lambda x: x[1], reverse=True)[:5]
            top5_btw = sorted(btw_cen.items(), key=lambda x: x[1], reverse=True)[:5]
            top5_eig = sorted(eig_cen.items(), key=lambda x: x[1], reverse=True)[:5] if eig_cen else []

            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown("<div class='mw-card'><h4>Degree Centrality</h4>", unsafe_allow_html=True)
                for acc, val in top5_deg:
                    st.caption(f"@{acc}: {val:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with cb:
                st.markdown("<div class='mw-card'><h4>Betweenness Centrality</h4>", unsafe_allow_html=True)
                for acc, val in top5_btw:
                    st.caption(f"@{acc}: {val:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with cc:
                st.markdown("<div class='mw-card'><h4>Eigenvector Centrality</h4>", unsafe_allow_html=True)
                if top5_eig:
                    for acc, val in top5_eig:
                        st.caption(f"@{acc}: {val:.3f}")
                else:
                    st.caption("Tidak dapat dihitung (graf tidak konvergen)")
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Lihat contoh edge list mentah"):
                st.dataframe(pd.DataFrame(edges[:20], columns=["source", "mention"]), use_container_width=True)

    # -------- TAB 5: DATA --------
    with tab5:
        st.subheader("📋 Tabel Data Lengkap")
        f1, f2 = st.columns(2)
        with f1:
            sent_filter = st.multiselect("Filter Sentimen:", df["sentiment"].unique(), default=list(df["sentiment"].unique()))
        with f2:
            urg_filter = st.multiselect("Filter Urgensi:", df["urgency_classification"].unique(),
                                         default=list(df["urgency_classification"].unique()))

        filtered = df[df["sentiment"].isin(sent_filter) & df["urgency_classification"].isin(urg_filter)]
        display_cols = [text_col, "sentiment", "sentiment_score", "urgency_classification",
                         "akar_masalah_label", "kmeans_cluster"]
        if username_col:
            display_cols.insert(0, username_col)
        st.dataframe(filtered[[c for c in display_cols if c in filtered.columns]],
                     use_container_width=True, height=400)

        csv_out = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Hasil Analisis (CSV)", csv_out,
                            "mindwatch_hasil.csv", "text/csv")

    # -------- TAB 6: ML CLASSIFIER --------
    with tab6:
        st.subheader("🤖 Machine Learning Klasifikasi (TF-IDF + Naive Bayes + SVM)")
        st.caption("Mengikuti notebook dosen: TF-IDF (max_features=1000) → train/test split 80/20 (stratified) → Naive Bayes & SVM Linear.")

        st.markdown("#### Klasifikasi Urgensi")
        st.write("Distribusi kelas:")
        st.write(df["urgency_classification"].value_counts())

        urgency_results = run_ml_classification(df["clean_text"], df["urgency_classification"])
        if urgency_results is None:
            st.info("⚠️ Data urgensi hanya memiliki 1 kelas pada dataset ini — model klasifikasi tidak bisa dilatih (butuh minimal 2 kelas).")
        else:
            cols = st.columns(len(urgency_results))
            for col, (nama_model, res) in zip(cols, urgency_results.items()):
                with col:
                    st.metric(nama_model, f"{res['accuracy']*100:.2f}%")

        st.divider()
        st.markdown("#### Klasifikasi Sentimen")
        st.write("Distribusi kelas:")
        st.write(df["sentiment"].value_counts())

        sentiment_results = run_ml_classification(df["clean_text"], df["sentiment"])
        if sentiment_results is None:
            st.info("⚠️ Data sentimen hanya memiliki 1 kelas pada dataset ini — model klasifikasi tidak bisa dilatih (butuh minimal 2 kelas).")
        else:
            cols = st.columns(len(sentiment_results))
            for col, (nama_model, res) in zip(cols, sentiment_results.items()):
                with col:
                    st.metric(nama_model, f"{res['accuracy']*100:.2f}%")

    # ---- Footer ----
    st.markdown(f"""
    <div class="mw-footer">
    🌸 <b>MindWatch Dashboard</b> — Riset & Kesadaran Sosial Isu Kesehatan Mental Gen Z<br>
    {CRISIS_RESOURCES}
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="mw-card" style="text-align:center; padding:40px;">
    <h4>👈 Upload Dataset CSV di Sidebar</h4>
    <p style="color:#f0c0d8; font-size:14.5px; max-width:600px; margin: 12px auto;">
    Format mengikuti hasil crawling <b>tweet-harvest</b> (kolom: full_text, username, in_reply_to_screen_name, created_at, user_id_str).
    </p>
    </div>
    """, unsafe_allow_html=True)
