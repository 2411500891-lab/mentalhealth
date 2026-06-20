import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    classification_report, confusion_matrix, accuracy_score, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="MindWatch | Monitoring Isu Kesehatan Mental Gen Z",
    page_icon="🌷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN TOKENS — SOFT PASTEL PINK
# ============================================================
C_BG          = "#FFF9FB"   # putih hangat keunguan
C_CARD        = "#FFEEF3"   # card
C_ACCENT      = "#F7A8C4"   # pink pastel utama
C_ACCENT_LT   = "#FBC4D8"   # pink lebih muda
C_ACCENT_DUST = "#E893B5"   # pink dusty, teks aksen
C_MINT        = "#B8E3D8"   # positif / aman
C_PEACH       = "#FFD9A0"   # perhatian
C_SOFTRED     = "#F4A6A6"   # urgent (lembut)
C_TEXT        = "#5C4150"   # plum gelap lembut

PASTEL_SEQ = [C_ACCENT, C_MINT, C_PEACH, C_ACCENT_DUST, C_SOFTRED, C_ACCENT_LT]
COLOR_MAP_URGENCY = {
    "🚨 Darurat": C_SOFTRED,
    "⚠️ Perlu Perhatian": C_PEACH,
    "💬 Curhat Ringan": C_MINT,
}
PASTEL_SCALE = [[0.0, C_BG], [0.5, C_ACCENT_LT], [1.0, C_ACCENT_DUST]]

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Quicksand:wght@400;500;600;700&family=Nunito:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Nunito', sans-serif; color: {C_TEXT}; }}
h1, h2, h3, h4, h5 {{ font-family: 'Fredoka', sans-serif; color: {C_TEXT}; }}
.mw-mono {{ font-family: 'JetBrains Mono', monospace; }}

.stApp {{
    background: {C_BG};
    color: {C_TEXT};
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FFF3F7 0%, #FFE8EF 100%);
    border-right: 1px solid {C_ACCENT_LT};
}}
section[data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}

/* Hero */
.mw-hero {{
    padding: 30px 34px;
    border-radius: 26px;
    background: linear-gradient(120deg, {C_ACCENT_LT} 0%, #FFF3F7 55%, {C_ACCENT} 100%);
    border: 1px solid {C_ACCENT_LT};
    box-shadow: 0 12px 32px rgba(247,168,196,0.35);
    margin-bottom: 18px;
}}
.mw-hero h1 {{
    font-size: 30px; font-weight: 700; margin: 0 0 6px 0;
    color: {C_TEXT};
}}
.mw-hero p {{ color: {C_ACCENT_DUST}; font-size: 14.5px; margin: 0; font-weight: 600; }}
.mw-hero .mw-eyebrow {{
    display:inline-block; font-family:'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 1.5px; text-transform: uppercase; color: {C_ACCENT_DUST};
    background: rgba(255,255,255,0.55); padding: 3px 10px; border-radius: 999px; margin-bottom: 10px;
}}

/* Glass card */
.mw-card {{
    background: {C_CARD};
    border: 1px solid {C_ACCENT_LT};
    border-radius: 20px;
    padding: 18px 20px;
    height: 100%;
    margin-bottom: 12px;
}}
.mw-card h4 {{
    margin-top: 0; font-size: 13px; color: {C_ACCENT_DUST};
    font-weight: 700; letter-spacing: .4px; text-transform: uppercase; font-family:'Quicksand',sans-serif;
}}
.mw-card .big {{ font-size: 30px; font-weight: 700; color: {C_ACCENT_DUST}; font-family:'Fredoka',sans-serif; }}
.mw-card p {{ color: {C_TEXT}; }}

/* Insight box */
.mw-insight {{
    background: linear-gradient(120deg, rgba(184,227,216,0.35), rgba(247,168,196,0.18));
    border: 1px dashed {C_ACCENT_DUST};
    border-radius: 18px;
    padding: 16px 20px;
    margin: 14px 0;
}}
.mw-insight h4 {{ margin-top:0; font-size:13px; color:{C_ACCENT_DUST}; text-transform:uppercase; letter-spacing:.4px; }}
.mw-insight ul {{ margin: 6px 0 0 0; padding-left: 18px; }}
.mw-insight li {{ font-size: 13.5px; line-height: 1.7; color: {C_TEXT}; }}

/* Pipeline step card */
.mw-step {{
    background: {C_CARD};
    border: 1px solid {C_ACCENT_LT};
    border-radius: 16px;
    padding: 14px 16px;
    text-align: center;
    height: 100%;
}}
.mw-step .mw-step-no {{
    font-family:'JetBrains Mono', monospace; font-size: 11px; color:{C_ACCENT_DUST};
    background: {C_ACCENT_LT}; display:inline-block; padding:2px 9px; border-radius:999px; margin-bottom:6px;
}}
.mw-step .mw-step-title {{ font-family:'Fredoka',sans-serif; font-size:14px; color:{C_TEXT}; font-weight:600; }}
.mw-step .mw-step-desc {{ font-size: 11.5px; color:{C_ACCENT_DUST}; margin-top:2px; }}

.mw-before-after {{
    background: #fff; border: 1px solid {C_ACCENT_LT}; border-radius: 12px; padding: 10px 14px;
    font-family:'JetBrains Mono', monospace; font-size: 12px; color:{C_TEXT}; word-break: break-word;
}}

/* Traffic lights */
.mw-light-wrap {{
    display: flex; align-items: center; justify-content: center; gap: 22px;
    background: {C_CARD};
    border: 1px solid {C_ACCENT_LT};
    border-radius: 22px; padding: 22px;
}}
.mw-light {{ width: 46px; height: 46px; border-radius: 50%; opacity: 0.18; }}
.mw-light.on-red    {{ background: {C_SOFTRED}; opacity: 1; box-shadow: 0 0 30px 8px rgba(244,166,166,0.65); animation: pulse 1.6s infinite; }}
.mw-light.on-yellow {{ background: {C_PEACH}; opacity: 1; box-shadow: 0 0 30px 8px rgba(255,217,160,0.6); }}
.mw-light.on-green  {{ background: {C_MINT}; opacity: 1; box-shadow: 0 0 26px 6px rgba(184,227,216,0.6); }}
@keyframes pulse {{ 0%{{transform:scale(1);}} 50%{{transform:scale(1.12);}} 100%{{transform:scale(1);}} }}

.mw-status-text {{ font-family: 'Fredoka', sans-serif; font-weight: 600; font-size: 20px; color: {C_ACCENT_DUST}; text-align: center; margin-top: 10px; }}

/* Alert banner */
.mw-alert-banner {{
    border-radius: 18px; padding: 16px 20px; margin: 14px 0;
    background: linear-gradient(90deg, rgba(244,166,166,0.25), rgba(244,166,166,0.06));
    border: 1px solid {C_SOFTRED};
    font-size: 14px; line-height: 1.6; color: {C_TEXT};
}}
.mw-alert-banner b {{ color: #C9576E; }}

/* Metrics */
div[data-testid="stMetricValue"] {{ color: {C_ACCENT_DUST} !important; font-family:'JetBrains Mono', monospace !important; }}
div[data-testid="stMetric"] {{
    background: {C_CARD};
    border: 1px solid {C_ACCENT_LT};
    border-radius: 16px; padding: 10px 14px;
}}
div[data-testid="stMetricLabel"] p {{ color: {C_ACCENT_DUST} !important; font-weight:600; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; background: transparent; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{
    background: {C_CARD};
    border-radius: 14px 14px 0 0;
    padding: 8px 16px; color: {C_ACCENT_DUST};
    border: 1px solid {C_ACCENT_LT};
    font-family: 'Quicksand', sans-serif; font-weight: 600;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(90deg, {C_ACCENT}, {C_ACCENT_LT}) !important;
    color: #fff !important; border-color: {C_ACCENT} !important;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {C_ACCENT}, {C_ACCENT_DUST});
    color: white; border: none; border-radius: 14px;
    font-family: 'Fredoka', sans-serif; font-weight: 600;
    padding: 8px 20px; transition: all 0.2s;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {C_ACCENT_DUST}, {C_ACCENT});
    transform: translateY(-1px); box-shadow: 0 4px 18px rgba(232,147,181,0.4);
}}

/* Selectbox & Multiselect */
.stSelectbox > div > div, .stMultiSelect > div > div {{
    background: #fff !important;
    border: 1px solid {C_ACCENT_LT} !important;
    border-radius: 12px !important; color: {C_TEXT} !important;
}}

/* Text area */
.stTextArea > div > div > textarea {{
    background: #fff !important;
    border: 1px solid {C_ACCENT_LT} !important;
    border-radius: 14px !important; color: {C_TEXT} !important;
}}

/* File uploader */
[data-testid="stFileUploader"] {{
    background: {C_CARD};
    border: 1.5px dashed {C_ACCENT};
    border-radius: 16px; padding: 12px;
}}

hr {{ border-color: {C_ACCENT_LT} !important; }}

[data-testid="stDataFrame"] {{ border: 1px solid {C_ACCENT_LT} !important; border-radius: 14px !important; }}

.mw-footer {{
    margin-top: 30px; padding: 18px; border-radius: 18px;
    background: {C_CARD}; border: 1px solid {C_ACCENT_LT};
    font-size: 12.5px; color: {C_ACCENT_DUST}; text-align: center;
}}

.mw-badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px; font-family:'Quicksand',sans-serif; }}
.badge-urgent {{ background:rgba(244,166,166,0.3); color:#C9576E; border:1px solid {C_SOFTRED}; }}
.badge-watch  {{ background:rgba(255,217,160,0.4); color:#B97A1E; border:1px solid {C_PEACH}; }}
.badge-mild   {{ background:rgba(184,227,216,0.4); color:#2E8C76; border:1px solid {C_MINT}; }}

.mw-mono-val {{ font-family:'JetBrains Mono', monospace; font-size: 13px; color:{C_TEXT}; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# KAMUS & KONSTANTA
# ============================================================
NORMALIZATION_DICT = {
    "yg":"yang","ga":"tidak","gak":"tidak","gk":"tidak","tdk":"tidak","engga":"tidak","nggak":"tidak",
    "udh":"sudah","udah":"sudah","sdh":"sudah","dah":"sudah","blm":"belum","blom":"belum",
    "gw":"aku","gue":"aku","gua":"aku","w":"aku","sy":"saya","aq":"aku",
    "lu":"kamu","lo":"kamu","loe":"kamu","km":"kamu",
    "bgt":"banget","bngt":"banget","bgttt":"banget",
    "tp":"tapi","krn":"karena","krna":"karena","jd":"jadi","jdi":"jadi",
    "knp":"kenapa","gmn":"gimana","emg":"memang","emang":"memang",
    "skrg":"sekarang","skg":"sekarang","td":"tadi",
    "org":"orang","ortu":"orang tua","klrg":"keluarga",
    "capek":"lelah","cape":"lelah","cpe":"lelah",
    "sdih":"sedih","nangis":"menangis","nangis2":"menangis",
    "depresi":"depresi","anxiety":"cemas","anxious":"cemas",
    "overthink":"cemas berlebihan","overthinking":"cemas berlebihan",
    "burnout":"burnout","insecure":"tidak percaya diri",
    "moodswing":"perubahan mood","gabut":"bosan","mager":"malas",
    "circle":"lingkar pertemanan","baper":"terbawa perasaan",
    "ghosting":"diabaikan","php":"diberi harapan palsu",
    "toxic":"toksik","gaslighting":"manipulasi psikologis",
    "deadline":"tenggat waktu","dl":"tenggat waktu",
    "dospem":"dosen pembimbing","duit":"uang","ukt":"biaya kuliah",
    "thx":"terima kasih","makasih":"terima kasih","mksh":"terima kasih",
    "semngt":"semangat","smngt":"semangat","semangatt":"semangat",
}

STOPWORDS = {
    "yang","di","dan","itu","ini","dari","ke","untuk","dengan","nya","saya","aku","kamu",
    "kami","kita","bisa","ada","adalah","juga","karena","tapi","namun","atau","jadi",
    "jika","kalau","sudah","lagi","akan","pada","masih","saja","yg","dg","dgn","ny","d","k",
    "biar","bikin","bilang","nih","sih","si","tau","tuh","ya","jd","jgn","aja","n","t",
    "loh","oleh","se","an","kan","dia","mereka","ia","telah","sedang","pernah","belum",
    "bukan","jangan","bila","maka","dalam","kepada","terhadap","antara","tentang",
    "hingga","sambil","demi","sebelum","sesudah","saat","ketika","begitu","seperti",
    "secara","setiap","seluruh","semua","para","sang","deh","dong","kok","lho","kah","pun","banget",
    "http","co","amp","rt","via","the","is","in","to","a","of","for","and","on",
}

KEEP_WORDS = {"tidak"}

SEVERE_URGENT_PHRASES = {
    "bunuh diri","mengakhiri hidup","pengen mati","ingin mati","udah cape hidup",
    "ga sanggup hidup","tidak sanggup hidup","self harm","melukai diri","nyakitin diri",
    "tidak ada gunanya hidup","gapunya alasan hidup","mau menyerah sama hidup",
    "pengen ngilang selamanya","capek jadi beban","mending aku ga ada",
    "udah ga kuat lagi hidup","pengen hilang aja","mau pergi selamanya",
    "menyerah","pengen menyerah",
}

MODERATE_URGENT_PHRASES = {
    "depresi","cemas berlebihan","panic attack","serangan panik","insomnia parah",
    "ga kuat lagi","tidak kuat lagi","trauma","burnout","stress berat","stres berat",
    "sendirian banget","butuh pertolongan","butuh bantuan","gapunya siapa siapa",
    "hopeless","putus asa","kosong banget","hampa banget","numb","mental breakdown",
    "nangis terus","susah tidur","ga nafsu makan","tidak nafsu makan","menyakiti diri",
    "anxiety","gangguan kecemasan","gangguan mental","kesehatan mental",
}

MILD_PHRASES = {
    "lelah","stress","stres","butuh teman cerita","pengen cerita","kuliah berat",
    "tugas menumpuk","tenggat waktu","tidak percaya diri","terbawa perasaan",
    "perubahan mood","bosan","malas","kesepian","capek","kelelahan","sedih",
    "khawatir","galau","overthinking","overthink",
}

CLUSTER_KEYWORDS = {
    "Tekanan Akademik": {"tugas","skripsi","ujian","dosen","kuliah","tenggat waktu","ipk","krs",
                          "sidang","praktikum","semester","kampus","magang","nilai","revisi","sks"},
    "Masalah Keluarga": {"orang tua","ayah","ibu","keluarga","broken home","kdrt",
                          "cerai","dibanding bandingkan","tekanan keluarga","ekspektasi",
                          "berantem","dipaksa"},
    "Finansial": {"uang","biaya kuliah","biaya","ekonomi","kerja","gaji","utang","beasiswa",
                  "finansial","susah cari kerja","tidak punya uang","nunggak"},
    "Hubungan Sosial": {"pacar","putus","mantan","teman","lingkar pertemanan","sahabat",
                         "dikhianati","selingkuh","toksik","dijauhin","kesepian","diabaikan",
                         "diberi harapan palsu","manipulasi psikologis","bullying","dibully"},
    "Kesehatan Mental Umum": {"depresi","cemas","anxiety","stress","stres","burnout","trauma",
                               "insomnia","psikolog","psikiater","konseling","terapi","kesehatan jiwa"},
}

SUPPORT_PHRASES = {
    "semangat","kamu kuat","gpp","gapapa","tidak apa apa","ada aku","dm aja","cerita yuk",
    "pelukan","peluk","stay strong","kamu berharga","tetap semangat","jangan menyerah",
    "kita disini","sini cerita","semangat ya","kamu hebat","peluk jauh","kamu tidak sendirian",
    "boleh cerita","aku dengerin","semoga membaik","semoga lekas membaik","tetap kuat",
    "kamu pasti bisa","jangan lupa makan","jangan lupa istirahat","sehat selalu",
}

CRISIS_RESOURCES = """
📞 **SEJIWA (Kemenkes):** 119 ext 8  |  📞 **Into The Light:** intothelightid.org  |  📞 **Yayasan Pulih:** (021) 78842580
"""

PIPELINE_STEPS = [
    ("01", "Raw Text", "Teks asli dari dataset"),
    ("02", "Case Folding", "Menyamakan ke huruf kecil"),
    ("03", "Cleaning", "Hapus URL, mention, angka, tanda baca"),
    ("04", "Normalisasi Slang", "yg→yang, ga→tidak, gw→aku, dst"),
    ("05", "Tokenisasi", "Memecah kalimat menjadi token/kata"),
    ("06", "Stopword Removal", "Buang kata umum non-bermakna"),
    ("07", "Sastrawi Stemming", "Kembalikan kata ke bentuk dasar"),
]

# ============================================================
# PREPROCESSING — STAGE BY STAGE
# ============================================================
def case_fold(text):
    return text.lower() if isinstance(text, str) else ""

def clean_only(text):
    t = re.sub(r"http\S+|www\.\S+", " ", text)
    t = re.sub(r"@\w+|#\w+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def normalize_text(text):
    words = text.split()
    return " ".join(NORMALIZATION_DICT.get(w, w) for w in words)

def tokenize(text):
    return text.split()

def remove_stopwords_tokens(tokens):
    return [w for w in tokens if (w not in STOPWORDS or w in KEEP_WORDS) and len(w) > 2]

@st.cache_resource(show_spinner=False)
def get_stemmer():
    if SASTRAWI_AVAILABLE:
        factory = StemmerFactory()
        return factory.create_stemmer()
    return None

def stem_tokens(tokens, stemmer, cache):
    out = []
    for w in tokens:
        if w in cache:
            out.append(cache[w])
        else:
            s = stemmer.stem(w) if stemmer else w
            cache[w] = s
            out.append(s)
    return out

def clean_text(text):
    """Backward-compatible single-shot cleaner (case fold + clean)."""
    if not isinstance(text, str):
        return ""
    return clean_only(case_fold(text))

def remove_stopwords(text):
    return " ".join(remove_stopwords_tokens(text.split()))

@st.cache_data(show_spinner=False)
def run_full_pipeline(texts):
    """Run every preprocessing stage on a list of texts, returning a DataFrame of stages."""
    stemmer = get_stemmer()
    stem_cache = {}
    rows = []
    for raw in texts:
        raw = raw if isinstance(raw, str) else ""
        cf = case_fold(raw)
        cl = clean_only(cf)
        nm = normalize_text(cl)
        tk = tokenize(nm)
        sw = remove_stopwords_tokens(tk)
        st_ = stem_tokens(sw, stemmer, stem_cache)
        rows.append({
            "raw": raw,
            "case_folded": cf,
            "cleaned": cl,
            "normalized": nm,
            "tokenized": tk,
            "stopword_removed": " ".join(sw),
            "stemmed": " ".join(st_),
        })
    return pd.DataFrame(rows)

def preprocess(text):
    """Final combined preprocessing used for LDA / K-Means / ML features."""
    cf = case_fold(text if isinstance(text, str) else "")
    cl = clean_only(cf)
    nm = normalize_text(cl)
    sw = remove_stopwords_tokens(tokenize(nm))
    stemmer = get_stemmer()
    cache = st.session_state.setdefault("_stem_cache_single", {})
    st_ = stem_tokens(sw, stemmer, cache)
    return " ".join(st_)

def pipeline_word_stats(pipe_df):
    stages = ["raw", "case_folded", "cleaned", "normalized", "stopword_removed", "stemmed"]
    labels = ["Raw", "Case Folding", "Cleaning", "Normalisasi", "Stopword Removal", "Stemming"]
    rows = []
    for stage, label in zip(stages, labels):
        all_words = " ".join(pipe_df[stage]).split()
        rows.append({"Tahap": label, "Total Kata": len(all_words), "Kata Unik": len(set(all_words))})
    return pd.DataFrame(rows)

# ============================================================
# KLASIFIKASI URGENSI
# ============================================================
def score_urgency(raw_text_lower):
    severe = sum(1 for p in SEVERE_URGENT_PHRASES if p in raw_text_lower)
    moderate = sum(1 for p in MODERATE_URGENT_PHRASES if p in raw_text_lower)
    mild = sum(1 for p in MILD_PHRASES if p in raw_text_lower)
    score = severe * 3 + moderate * 1.5 + mild * 0.5

    if severe >= 1 or score >= 4:
        label = "🚨 Darurat"
    elif moderate >= 1 or score >= 1.5:
        label = "⚠️ Perlu Perhatian"
    else:
        label = "💬 Curhat Ringan"
    return label, score, severe, moderate, mild

def detect_cluster(raw_text_lower):
    scores = {c: sum(1 for kw in kws if kw in raw_text_lower) for c, kws in CLUSTER_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Lainnya"

def detect_support(raw_text_lower):
    return any(p in raw_text_lower for p in SUPPORT_PHRASES)

# ============================================================
# AUTO-DETEKSI KOLOM
# ============================================================
def auto_detect(df, candidates):
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return None

def detect_text_col(df):
    return auto_detect(df, ["text","tweet","full_text","content","caption","isi","teks","curhatan"])

def detect_username_col(df):
    return auto_detect(df, ["username","user","account","akun","screen_name","author","nama_user"])

def detect_parent_col(df):
    return auto_detect(df, ["in_reply_to_screen_name","in_reply_to_user","in_reply_to","reply_to","parent","membalas"])

def detect_timestamp_col(df):
    return auto_detect(df, ["created_at","timestamp","date","tanggal","waktu","time"])

def detect_engagement_cols(df):
    candidates = ["like_count","likes","favorite_count","retweet_count","retweets",
                  "reply_count","replies","quote_count","views","view_count"]
    found = [c for c in df.columns if c.lower().strip() in candidates]
    return found

# ============================================================
# LDA TOPIC MODELING
# ============================================================
@st.cache_data(show_spinner=False)
def run_lda(texts, n_topics=3):
    vectorizer_bow = CountVectorizer(max_features=1000)
    X_bow = vectorizer_bow.fit_transform(texts)
    terms = vectorizer_bow.get_feature_names_out()

    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=50)
    lda_model.fit(X_bow)

    doc_topic = lda_model.transform(X_bow)
    dominant_topics = doc_topic.argmax(axis=1)

    log_lik = lda_model.score(X_bow)
    perplexity = lda_model.perplexity(X_bow)

    topic_top_words = {}
    for k in range(n_topics):
        top_idx = lda_model.components_[k].argsort()[::-1][:8]
        topic_top_words[k] = [terms[i] for i in top_idx]

    return dominant_topics, topic_top_words, log_lik, perplexity, lda_model, vectorizer_bow

# ============================================================
# K-MEANS CLUSTERING
# ============================================================
@st.cache_data(show_spinner=False)
def run_kmeans(texts, k=3):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)

    wcss = []
    k_range = range(2, min(7, len(texts)))
    for ki in k_range:
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_2d)
        wcss.append(km.inertia_)

    km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_2d)

    sil = silhouette_score(X_2d, labels) if len(set(labels)) > 1 else 0
    dbi = davies_bouldin_score(X_2d, labels) if len(set(labels)) > 1 else 0
    chi = calinski_harabasz_score(X_2d, labels) if len(set(labels)) > 1 else 0

    return labels, X_2d, km_final, list(k_range), wcss, sil, dbi, chi

# ============================================================
# ML CLASSIFIER (TF-IDF + Logistic Regression, label = urgensi leksikon)
# ============================================================
@st.cache_data(show_spinner=False)
def train_ml_classifier(texts, labels):
    vectorizer = TfidfVectorizer(max_features=1500)
    X = vectorizer.fit_transform(texts)
    classes = sorted(set(labels))

    can_stratify = all(labels.count(c) >= 2 for c in classes) and len(classes) > 1
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.25, random_state=42,
            stratify=labels if can_stratify else None,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return model, vectorizer, classes, report, cm, acc, f1m, len(y_test)

# ============================================================
# SNA
# ============================================================
def build_network(df, text_col, username_col, parent_col):
    G = nx.DiGraph()
    mention_pattern = re.compile(r"@(\w+)")

    for idx, row in df.iterrows():
        raw = str(row[text_col])
        raw_lower = raw.lower()
        is_support = detect_support(raw_lower)

        source = (
            str(row[username_col]).strip()
            if username_col and pd.notna(row.get(username_col))
            else f"akun_{idx}"
        )
        targets = set()

        if parent_col and pd.notna(row.get(parent_col, None)):
            val = str(row[parent_col]).strip().lstrip("@")
            if val and val != "nan":
                targets.add(val)

        for m in mention_pattern.findall(raw):
            targets.add(m)

        for t in targets:
            if t == source or t == "":
                continue
            w = 2 if is_support else 1
            if G.has_edge(source, t):
                G[source][t]["weight"] += w
                if is_support:
                    G[source][t]["support"] = True
            else:
                G.add_edge(source, t, weight=w, support=is_support)

    return G

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

    pos = nx.spring_layout(G, k=0.6, seed=42)
    edge_x, edge_y, sup_x, sup_y = [], [], [], []

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        if d.get("support"):
            sup_x += [x0, x1, None]; sup_y += [y0, y1, None]
        else:
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.6, color="rgba(232,147,181,0.25)"), hoverinfo="none")
    sup_trace = go.Scatter(x=sup_x, y=sup_y, mode="lines",
                            line=dict(width=1.8, color="rgba(184,227,216,0.85)"), hoverinfo="none")

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
        marker=dict(size=node_sz, color=node_col, colorscale=[[0, C_ACCENT_LT], [1, C_ACCENT_DUST]],
                    line=dict(width=1, color="rgba(255,255,255,0.7)"), showscale=False)
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
# PLOTLY THEME HELPER
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color=C_TEXT,
    font_family="Nunito",
)

# ============================================================
# INSIGHT GENERATORS
# ============================================================
def insight_card(title, bullets):
    items = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f"""
    <div class="mw-insight">
    <h4>💡 {title}</h4>
    <ul>{items}</ul>
    </div>
    """, unsafe_allow_html=True)

def generate_urgency_insight(df):
    total = len(df)
    n_darurat = (df["urgensi"] == "🚨 Darurat").sum()
    n_perhatian = (df["urgensi"] == "⚠️ Perlu Perhatian").sum()
    pct_darurat = n_darurat / total * 100 if total else 0
    top_cluster = df["klaster_leksikon"].value_counts().idxmax() if total else "-"
    top_cluster_pct = df["klaster_leksikon"].value_counts(normalize=True).max() * 100 if total else 0
    support_rate = df["is_support"].mean() * 100 if total else 0
    bullets = [
        f"<b>{pct_darurat:.1f}%</b> dari seluruh curhatan terklasifikasi darurat ({int(n_darurat)} dari {total} entri).",
        f"Akar masalah paling dominan: <b>{top_cluster}</b> ({top_cluster_pct:.1f}% dari total).",
        f"Sekitar <b>{int(n_perhatian)}</b> curhatan berada di level 'Perlu Perhatian' — kandidat intervensi pencegahan.",
        f"<b>{support_rate:.1f}%</b> dari seluruh teks mengandung respons/kalimat suportif terhadap sesama pengguna.",
    ]
    return bullets

def generate_lda_insight(df, topic_words, n_topics):
    dist = df["lda_topic"].value_counts(normalize=True) * 100
    dom_topic = dist.idxmax()
    bullets = [
        f"Topik paling banyak dibicarakan: <b>Topik {dom_topic}</b> ({', '.join(topic_words[dom_topic][:3])}) — {dist.max():.1f}% dokumen.",
        f"Model membagi curhatan ke dalam <b>{n_topics} topik laten</b> berdasarkan kombinasi kata yang sering muncul bersamaan.",
    ]
    if n_topics > 1:
        second = dist.sort_values(ascending=False).index[1]
        bullets.append(f"Topik kedua terbesar: <b>Topik {second}</b> ({', '.join(topic_words[second][:3])}) — {dist[second]:.1f}% dokumen.")
    return bullets

def generate_kmeans_insight(df, sil, n_clusters):
    sizes = df["kmeans_cluster"].value_counts(normalize=True) * 100
    biggest = sizes.idxmax()
    quality = "cukup baik" if sil >= 0.3 else ("sedang" if sil >= 0.1 else "lemah, klaster saling tumpang tindih")
    bullets = [
        f"Klaster terbesar adalah <b>Klaster {biggest}</b>, mencakup {sizes.max():.1f}% dari seluruh data.",
        f"Silhouette Score sebesar <b>{sil:.3f}</b> menunjukkan pemisahan klaster yang {quality}.",
        f"Data terbagi ke dalam <b>{n_clusters} klaster</b> berdasarkan kemiripan kata pada representasi TF-IDF + SVD.",
    ]
    return bullets

def generate_sna_insight(G):
    if G.number_of_nodes() == 0:
        return ["Tidak ditemukan relasi mention/reply yang cukup untuk dianalisis."]
    n_sup_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("support"))
    density = nx.density(G)
    deg_cen = nx.degree_centrality(G)
    top_acc = max(deg_cen, key=deg_cen.get) if deg_cen else "-"
    bullets = [
        f"Jaringan terdiri dari <b>{G.number_of_nodes()} akun</b> dan <b>{G.number_of_edges()} interaksi</b>, dengan kepadatan jaringan {density:.4f}.",
        f"<b>{n_sup_edges}</b> dari seluruh interaksi adalah respons suportif terhadap curhatan orang lain.",
        f"Akun paling sentral dalam jaringan: <b>@{top_acc}</b>, kemungkinan menjadi simpul utama support system.",
    ]
    return bullets

# ============================================================
# UI — HERO
# ============================================================
st.markdown("""
<div class="mw-hero">
  <span class="mw-eyebrow">Social Listening · Kesehatan Mental</span>
  <h1>🌷 MindWatch — Monitoring Isu Kesehatan Mental Gen Z</h1>
  <p>Pipeline preprocessing · Statistik deskriptif · Deteksi urgensi · LDA · K-Means · ML Classifier · Social Network Analysis</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🌸 Upload Dataset")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    st.divider()
    st.markdown("### ⚙️ Parameter Analisis")
    n_topics = st.slider("Jumlah Topik LDA (K):", 2, 6, 3)
    n_clusters = st.slider("Jumlah Klaster K-Means:", 2, 6, 3)
    st.divider()
    if not SASTRAWI_AVAILABLE:
        st.warning("Paket **Sastrawi** tidak terpasang — tahap stemming akan dilewati (kata dibiarkan apa adanya). Jalankan `pip install Sastrawi` untuk mengaktifkan stemming penuh.")
    st.markdown("### ℹ️ Tentang")
    st.caption("**Tema 7** — Monitoring Isu Kesehatan Mental Gen Z\nKomponen: Preprocessing · Statistik · Urgensi · LDA · K-Means · ML Classifier · SNA")
    st.divider()
    st.markdown("### 🆘 Krisis? Hubungi:")
    st.caption(CRISIS_RESOURCES)

# ============================================================
# MAIN CONTENT
# ============================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset dimuat — **{len(df):,} baris**, **{len(df.columns)} kolom**")

    with st.expander("⚙️ Konfigurasi Kolom Dataset", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            default_text = detect_text_col(df)
            text_col = st.selectbox("Kolom Teks:", df.columns,
                                     index=list(df.columns).index(default_text) if default_text else 0)
            default_user = detect_username_col(df)
            username_col = st.selectbox("Kolom Username (opsional):", ["(tidak ada)"] + list(df.columns),
                                         index=(list(df.columns).index(default_user) + 1) if default_user else 0)
            username_col = None if username_col == "(tidak ada)" else username_col
        with c2:
            default_parent = detect_parent_col(df)
            parent_col = st.selectbox("Kolom Reply-To (opsional):", ["(tidak ada)"] + list(df.columns),
                                       index=(list(df.columns).index(default_parent) + 1) if default_parent else 0)
            parent_col = None if parent_col == "(tidak ada)" else parent_col
            default_ts = detect_timestamp_col(df)
            ts_col = st.selectbox("Kolom Waktu (opsional):", ["(tidak ada)"] + list(df.columns),
                                   index=(list(df.columns).index(default_ts) + 1) if default_ts else 0)
            ts_col = None if ts_col == "(tidak ada)" else ts_col

    with st.spinner("🌸 Memproses data..."):
        df["text_clean"] = df[text_col].astype(str).str.lower()

        pipe_df = run_full_pipeline(df[text_col].astype(str).tolist())
        df["text_preprocessed"] = pipe_df["stemmed"].values

        urgency_results = df["text_clean"].apply(score_urgency)
        df["urgensi"]       = urgency_results.apply(lambda x: x[0])
        df["skor_urgensi"]  = urgency_results.apply(lambda x: x[1])
        df["n_severe"]      = urgency_results.apply(lambda x: x[2])
        df["n_moderate"]    = urgency_results.apply(lambda x: x[3])
        df["n_mild"]        = urgency_results.apply(lambda x: x[4])
        df["klaster_leksikon"] = df["text_clean"].apply(detect_cluster)
        df["is_support"]    = df["text_clean"].apply(detect_support)
        df["jumlah_kata"]   = df[text_col].astype(str).apply(lambda t: len(t.split()))
        df["jumlah_karakter"] = df[text_col].astype(str).apply(len)

        if ts_col:
            df["_ts_parsed"] = pd.to_datetime(df[ts_col], errors="coerce")

        engagement_cols = detect_engagement_cols(df)

        clean_texts = df["text_preprocessed"].tolist()
        dom_topics, topic_words, log_lik, perplexity, lda_model, bow_vec = run_lda(clean_texts, n_topics)
        df["lda_topic"] = dom_topics
        topic_label_map = {i: f"Topik {i}: {', '.join(topic_words[i][:3])}" for i in range(n_topics)}
        df["lda_topic_label"] = df["lda_topic"].map(topic_label_map)

        km_labels, X_2d, km_final, k_range, wcss, sil, dbi, chi = run_kmeans(clean_texts, n_clusters)
        df["kmeans_cluster"] = km_labels

        G = build_network(df, text_col, username_col, parent_col)

    total = len(df)
    n_darurat   = (df["urgensi"] == "🚨 Darurat").sum()
    n_perhatian = (df["urgensi"] == "⚠️ Perlu Perhatian").sum()
    n_ringan    = (df["urgensi"] == "💬 Curhat Ringan").sum()
    pct_darurat = n_darurat / total * 100 if total else 0

    if pct_darurat >= 15:
        light_class, status_text = "on-red", "⚠️ SIAGA TINGGI"
    elif pct_darurat >= 5:
        light_class, status_text = "on-yellow", "👁️ PERLU DIPANTAU"
    else:
        light_class, status_text = "on-green", "✅ TERKENDALI"

    st.markdown("### 🚦 Indikator Urgensi")
    col_light, col_kpi = st.columns([1, 2.5])

    with col_light:
        st.markdown(f"""
        <div class="mw-light-wrap">
          <div class="mw-light" style="background:{C_SOFTRED}; opacity:{1 if light_class=='on-red' else 0.15}; {'box-shadow:0 0 30px 8px rgba(244,166,166,0.65);animation:pulse 1.6s infinite;' if light_class=='on-red' else ''}"></div>
          <div class="mw-light" style="background:{C_PEACH}; opacity:{1 if light_class=='on-yellow' else 0.15}; {'box-shadow:0 0 30px 8px rgba(255,217,160,0.6);' if light_class=='on-yellow' else ''}"></div>
          <div class="mw-light" style="background:{C_MINT}; opacity:{1 if light_class=='on-green' else 0.15}; {'box-shadow:0 0 26px 6px rgba(184,227,216,0.6);' if light_class=='on-green' else ''}"></div>
        </div>
        <p class="mw-status-text">{status_text}</p>
        """, unsafe_allow_html=True)

    with col_kpi:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Tweet", f"{total:,}")
        k2.metric("🚨 Darurat", int(n_darurat), f"{pct_darurat:.1f}%")
        k3.metric("⚠️ Perlu Perhatian", int(n_perhatian))
        k4.metric("💬 Curhat Ringan", int(n_ringan))

    if n_darurat > 0:
        st.markdown(f"""
        <div class="mw-alert-banner">
        <b>⚠️ Peringatan:</b> Terdeteksi <b>{int(n_darurat)} curhatan berisiko tinggi</b>.
        Segmen ini memerlukan perhatian segera.<br><small>{CRISIS_RESOURCES}</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ============================================================
    # TABS
    # ============================================================
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🧪 Preprocessing Pipeline",
        "📐 Statistik Deskriptif",
        "📈 Sentimen & Urgensi",
        "🔬 LDA Topic Modeling",
        "🧩 K-Means Clustering",
        "🤖 ML Classifier",
        "🕸️ Social Network (SNA)",
        "📋 Data & Export",
    ])

    # -------- TAB 0: PREPROCESSING PIPELINE --------
    with tab0:
        st.subheader("🧪 Tahapan Preprocessing Teks")
        st.caption("Setiap curhatan mentah melewati 7 tahap sebelum dipakai untuk pemodelan.")

        step_cols = st.columns(len(PIPELINE_STEPS))
        for col, (no, title, desc) in zip(step_cols, PIPELINE_STEPS):
            with col:
                st.markdown(f"""
                <div class="mw-step">
                <span class="mw-step-no">{no}</span><br>
                <span class="mw-step-title">{title}</span>
                <div class="mw-step-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### 🔍 Contoh Nyata dari Dataset")
        sample_idx = st.slider("Pilih baris dataset sebagai contoh:", 0, max(len(pipe_df) - 1, 0), 0)
        sample = pipe_df.iloc[sample_idx]

        ex_labels = [
            ("Raw Text", sample["raw"]),
            ("Case Folding", sample["case_folded"]),
            ("Cleaning (URL/mention/angka/tanda baca)", sample["cleaned"]),
            ("Normalisasi Slang", sample["normalized"]),
            ("Tokenisasi", " | ".join(sample["tokenized"])),
            ("Stopword Removal", sample["stopword_removed"]),
            ("Sastrawi Stemming (hasil akhir)", sample["stemmed"]),
        ]
        for label, val in ex_labels:
            st.markdown(f"**{label}**")
            st.markdown(f'<div class="mw-before-after">{val if val else "(kosong)"}</div>', unsafe_allow_html=True)
            st.write("")

        st.markdown("#### 📊 Statistik Kata Sebelum vs Sesudah Tiap Tahap")
        stats_df = pipeline_word_stats(pipe_df)
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        with c2:
            fig_funnel = go.Figure()
            fig_funnel.add_trace(go.Bar(x=stats_df["Tahap"], y=stats_df["Total Kata"],
                                         name="Total Kata", marker_color=C_ACCENT))
            fig_funnel.add_trace(go.Bar(x=stats_df["Tahap"], y=stats_df["Kata Unik"],
                                         name="Kata Unik", marker_color=C_MINT))
            fig_funnel.update_layout(**PLOTLY_LAYOUT, barmode="group", xaxis_title="", yaxis_title="Jumlah Kata")
            st.plotly_chart(fig_funnel, use_container_width=True)

        insight_card("Insight Preprocessing", [
            f"Tahap <b>cleaning</b> memangkas noise (URL, mention, angka, simbol) sebelum normalisasi dilakukan.",
            f"Setelah stopword removal, jumlah kata unik berkurang dari <b>{stats_df.loc[stats_df['Tahap']=='Normalisasi','Kata Unik'].values[0]}</b> menjadi <b>{stats_df.loc[stats_df['Tahap']=='Stopword Removal','Kata Unik'].values[0]}</b>.",
            "Stemming Sastrawi menyatukan variasi morfologis kata (mis. 'menangis' & 'tangisan') ke akar kata yang sama, sehingga representasi fitur TF-IDF/BoW lebih ringkas." if SASTRAWI_AVAILABLE else "Stemming dilewati karena paket Sastrawi tidak terdeteksi di environment ini.",
        ])

    # -------- TAB 1: STATISTIK DESKRIPTIF --------
    with tab1:
        st.subheader("📐 Statistik Deskriptif Dataset")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rata-rata Kata/Teks", f"{df['jumlah_kata'].mean():.1f}")
        c2.metric("Median Kata/Teks", f"{df['jumlah_kata'].median():.0f}")
        c3.metric("Rata-rata Karakter", f"{df['jumlah_karakter'].mean():.0f}")
        c4.metric("Teks Terpanjang (kata)", f"{df['jumlah_kata'].max()}")

        st.markdown("#### Distribusi Panjang Teks")
        c1, c2 = st.columns(2)
        with c1:
            fig_wc_len = px.histogram(df, x="jumlah_kata", nbins=30, color_discrete_sequence=[C_ACCENT])
            fig_wc_len.update_layout(**PLOTLY_LAYOUT, xaxis_title="Jumlah Kata", yaxis_title="Frekuensi")
            st.plotly_chart(fig_wc_len, use_container_width=True)
        with c2:
            fig_ch_len = px.histogram(df, x="jumlah_karakter", nbins=30, color_discrete_sequence=[C_MINT])
            fig_ch_len.update_layout(**PLOTLY_LAYOUT, xaxis_title="Jumlah Karakter", yaxis_title="Frekuensi")
            st.plotly_chart(fig_ch_len, use_container_width=True)

        if ts_col and df["_ts_parsed"].notna().sum() > 0:
            st.markdown("#### Distribusi Waktu Posting")
            tdf = df.dropna(subset=["_ts_parsed"]).copy()
            tdf["jam"] = tdf["_ts_parsed"].dt.hour
            tdf["tanggal"] = tdf["_ts_parsed"].dt.date.astype(str)
            c1, c2 = st.columns(2)
            with c1:
                hour_count = tdf["jam"].value_counts().sort_index().reset_index()
                hour_count.columns = ["Jam", "Jumlah"]
                fig_hour = px.bar(hour_count, x="Jam", y="Jumlah", color_discrete_sequence=[C_ACCENT_DUST])
                fig_hour.update_layout(**PLOTLY_LAYOUT, xaxis_title="Jam (0-23)", yaxis_title="Jumlah Postingan")
                st.plotly_chart(fig_hour, use_container_width=True)
            with c2:
                date_count = tdf["tanggal"].value_counts().sort_index().reset_index()
                date_count.columns = ["Tanggal", "Jumlah"]
                fig_date = px.line(date_count, x="Tanggal", y="Jumlah", markers=True,
                                    color_discrete_sequence=[C_ACCENT])
                fig_date.update_layout(**PLOTLY_LAYOUT, xaxis_title="Tanggal", yaxis_title="Jumlah Postingan")
                st.plotly_chart(fig_date, use_container_width=True)
        else:
            st.info("💡 Pilih kolom waktu di konfigurasi kolom untuk melihat distribusi jam/tanggal posting.")

        st.markdown("#### Kata Paling Sering Muncul (Sebelum & Sesudah Preprocessing)")
        c1, c2 = st.columns(2)
        with c1:
            raw_words = " ".join(pipe_df["cleaned"]).split()
            raw_top = pd.DataFrame(Counter(raw_words).most_common(15), columns=["Kata", "Frekuensi"])
            fig_raw_top = px.bar(raw_top.sort_values("Frekuensi"), x="Frekuensi", y="Kata", orientation="h",
                                  color="Frekuensi", color_continuous_scale=PASTEL_SCALE, title="Top Words — Raw (sebelum stopword removal)")
            fig_raw_top.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_raw_top, use_container_width=True)
        with c2:
            final_words = " ".join(df["text_preprocessed"]).split()
            if final_words:
                wc = WordCloud(width=900, height=420, background_color="white", mode="RGB",
                                colormap="RdPu", max_words=80).generate(" ".join(final_words))
                fig_wc, ax_wc = plt.subplots(figsize=(10, 4.5))
                fig_wc.patch.set_alpha(0)
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("Tidak cukup data untuk WordCloud.")

        if engagement_cols:
            st.markdown("#### Statistik Engagement")
            eng_summary = df[engagement_cols].describe().T[["mean", "max", "sum"]].reset_index()
            eng_summary.columns = ["Metrik", "Rata-rata", "Maksimum", "Total"]
            st.dataframe(eng_summary, use_container_width=True, hide_index=True)
            top_col = engagement_cols[0]
            top_eng = df.sort_values(top_col, ascending=False).head(5)[[text_col, top_col, "urgensi"]]
            st.markdown(f"**Top 5 teks dengan {top_col} tertinggi:**")
            st.dataframe(top_eng, use_container_width=True, hide_index=True)
        else:
            st.info("💡 Tidak ada kolom engagement (likes/retweet/replies) terdeteksi di dataset ini.")

    # -------- TAB 2: SENTIMEN & URGENSI --------
    with tab2:
        st.subheader("Distribusi Tingkat Urgensi")
        c1, c2 = st.columns(2)
        with c1:
            urg_count = df["urgensi"].value_counts().reset_index()
            urg_count.columns = ["Urgensi", "Jumlah"]
            fig_pie = px.pie(urg_count, names="Urgensi", values="Jumlah", hole=0.55,
                              color="Urgensi", color_discrete_map=COLOR_MAP_URGENCY)
            fig_pie.update_layout(**PLOTLY_LAYOUT, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_bar = px.bar(urg_count, x="Urgensi", y="Jumlah", color="Urgensi",
                              color_discrete_map=COLOR_MAP_URGENCY, text="Jumlah")
            fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Klaster Akar Masalah (Leksikon)")
        klaster_count = df["klaster_leksikon"].value_counts().reset_index()
        klaster_count.columns = ["Klaster", "Jumlah"]
        c3, c4 = st.columns(2)
        with c3:
            fig_kp = px.pie(klaster_count, names="Klaster", values="Jumlah", hole=0.45,
                             color_discrete_sequence=PASTEL_SEQ)
            fig_kp.update_layout(**PLOTLY_LAYOUT, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_kp, use_container_width=True)
        with c4:
            fig_kb = px.bar(klaster_count.sort_values("Jumlah"), x="Jumlah", y="Klaster",
                             orientation="h", color="Jumlah", color_continuous_scale=PASTEL_SCALE)
            fig_kb.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_kb, use_container_width=True)

        if ts_col and df["_ts_parsed"].notna().sum() > 0:
            st.subheader("Tren Urgensi dari Waktu ke Waktu")
            trend_df = df.dropna(subset=["_ts_parsed"]).copy()
            trend_df["periode"] = trend_df["_ts_parsed"].dt.to_period("D").astype(str)
            trend_pivot = trend_df.groupby(["periode", "urgensi"]).size().unstack(fill_value=0).reset_index()
            fig_trend = go.Figure()
            for label, color in COLOR_MAP_URGENCY.items():
                if label in trend_pivot.columns:
                    fig_trend.add_trace(go.Scatter(
                        x=trend_pivot["periode"], y=trend_pivot[label],
                        mode="lines+markers", name=label, line=dict(color=color, width=2.5)
                    ))
            fig_trend.update_layout(**PLOTLY_LAYOUT, xaxis_title="Tanggal", yaxis_title="Jumlah")
            st.plotly_chart(fig_trend, use_container_width=True)

        insight_card("Insight Otomatis — Urgensi", generate_urgency_insight(df))

    # -------- TAB 3: LDA --------
    with tab3:
        st.subheader("🔬 Latent Dirichlet Allocation (LDA) Topic Modeling")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="mw-card">
            <h4>Metrik Evaluasi Model LDA</h4>
            <p class="mw-mono-val">
            📊 <b>Log-Likelihood:</b> {log_lik:.2f}<br>
            📉 <b>Perplexity:</b> {perplexity:.2f} <small>(lebih kecil = lebih baik)</small><br>
            🔢 <b>Jumlah Topik:</b> {n_topics}
            </p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="mw-card">
            <h4>Cara Kerja LDA</h4>
            <p style="font-size:13px;">
            LDA (Latent Dirichlet Allocation) adalah model probabilistik yang mengasumsikan
            setiap dokumen merupakan campuran dari beberapa topik tersembunyi,
            dan setiap topik dicirikan oleh distribusi kata-kata.
            Menggunakan representasi <b>Bag of Words (BoW)</b> sesuai teori.
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Kata Kunci Tiap Topik")
        cols_topics = st.columns(n_topics)
        for i, col in enumerate(cols_topics):
            with col:
                words_str = " · ".join(topic_words[i])
                st.markdown(f"""
                <div class="mw-card">
                <h4>Topik {i}</h4>
                <p style="color:{C_ACCENT_DUST}; font-size:13px; line-height:1.8;">{words_str}</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("Distribusi Dokumen per Topik")
        topic_dist = df["lda_topic"].value_counts().reset_index()
        topic_dist.columns = ["Topik", "Jumlah"]
        topic_dist["Label"] = topic_dist["Topik"].map(lambda x: f"Topik {x}: {', '.join(topic_words[x][:2])}")
        fig_lda = px.bar(topic_dist, x="Label", y="Jumlah",
                          color="Jumlah", color_continuous_scale=PASTEL_SCALE, text="Jumlah")
        fig_lda.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, xaxis_title="", yaxis_title="Jumlah Dokumen")
        st.plotly_chart(fig_lda, use_container_width=True)

        if ts_col and df["_ts_parsed"].notna().sum() > 0:
            st.subheader("Tren Topik LDA dari Waktu ke Waktu")
            trend_lda = df.dropna(subset=["_ts_parsed"]).copy()
            trend_lda["periode"] = trend_lda["_ts_parsed"].dt.to_period("D").astype(str)
            trend_lda_pivot = trend_lda.groupby(["periode", "lda_topic"]).size().unstack(fill_value=0)
            trend_lda_pivot.columns = [f"Topik {c}" for c in trend_lda_pivot.columns]
            trend_lda_pivot = trend_lda_pivot.reset_index()

            fig_lda_trend = go.Figure()
            for i, col in enumerate([c for c in trend_lda_pivot.columns if c != "periode"]):
                fig_lda_trend.add_trace(go.Scatter(
                    x=trend_lda_pivot["periode"], y=trend_lda_pivot[col],
                    mode="lines+markers", name=col,
                    line=dict(color=PASTEL_SEQ[i % len(PASTEL_SEQ)], width=2.5),
                ))
            fig_lda_trend.update_layout(**PLOTLY_LAYOUT, xaxis_title="Tanggal", yaxis_title="Volume Isu")
            st.plotly_chart(fig_lda_trend, use_container_width=True)
        else:
            st.info("💡 Pilih kolom waktu di konfigurasi untuk melihat tren topik harian.")

        insight_card("Insight Otomatis — LDA", generate_lda_insight(df, topic_words, n_topics))

    # -------- TAB 4: K-MEANS --------
    with tab4:
        st.subheader("🧩 K-Means Clustering")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="mw-card">
            <h4>Metrik Evaluasi K-Means (K={n_clusters})</h4>
            <p class="mw-mono-val">
            🔵 <b>Silhouette Score:</b> {sil:.4f} <small>(mendekati 1 = baik)</small><br>
            📉 <b>Davies-Bouldin Index:</b> {dbi:.4f} <small>(makin kecil = makin baik)</small><br>
            📈 <b>Calinski-Harabasz:</b> {chi:.2f} <small>(makin besar = makin baik)</small>
            </p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="mw-card">
            <h4>Metode & Pipeline</h4>
            <p style="font-size:13px;">
            1️⃣ <b>TF-IDF Vectorizer</b> → representasi fitur teks<br>
            2️⃣ <b>TruncatedSVD (LSI)</b> → reduksi dimensi ke 2D<br>
            3️⃣ <b>K-Means</b> → clustering dengan K={n_clusters}<br>
            4️⃣ <b>Elbow Method</b> → penentuan K optimal
            </p>
            </div>
            """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Elbow Method")
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=list(k_range), y=wcss, mode="lines+markers",
                                            line=dict(color=C_ACCENT, width=2.5),
                                            marker=dict(size=8, color=C_ACCENT_DUST)))
            fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color=C_ACCENT_DUST,
                                 annotation_text=f"K={n_clusters}", annotation_font_color=C_ACCENT_DUST)
            fig_elbow.update_layout(**PLOTLY_LAYOUT, xaxis_title="Jumlah Klaster (K)", yaxis_title="WCSS (Inertia)")
            st.plotly_chart(fig_elbow, use_container_width=True)

        with c2:
            st.subheader("Visualisasi Klaster (2D)")
            df_plot = pd.DataFrame({"x": X_2d[:, 0], "y": X_2d[:, 1], "Klaster": df["kmeans_cluster"].astype(str)})
            centroids = km_final.cluster_centers_
            fig_scatter = px.scatter(df_plot, x="x", y="y", color="Klaster",
                                      color_discrete_sequence=PASTEL_SEQ,
                                      labels={"x": "SVD Komponen 1", "y": "SVD Komponen 2"})
            fig_scatter.add_trace(go.Scatter(
                x=centroids[:, 0], y=centroids[:, 1], mode="markers",
                marker=dict(symbol="x", size=16, color=C_TEXT, line=dict(width=2, color="white")),
                name="Centroid",
            ))
            fig_scatter.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Kata Dominan per Klaster (Top-5 Words)")
        cols_km = st.columns(n_clusters)
        for cluster_id in range(n_clusters):
            with cols_km[cluster_id]:
                cluster_texts = df[df["kmeans_cluster"] == cluster_id]["text_preprocessed"]
                all_words = " ".join(cluster_texts).split()
                word_counts = Counter(all_words)
                top_words = [w for w, _ in word_counts.most_common(5)]
                n_docs = len(cluster_texts)
                words_str = " · ".join(top_words) if top_words else "—"
                st.markdown(f"""
                <div class="mw-card">
                <h4>Klaster {cluster_id}</h4>
                <p style="font-size:12px;">📄 {n_docs} dokumen</p>
                <p style="color:{C_ACCENT_DUST}; font-size:13px; line-height:1.8;">{words_str}</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("WordCloud per Klaster")
        cluster_pick = st.selectbox("Pilih Klaster:", [f"Klaster {i}" for i in range(n_clusters)])
        cluster_id_pick = int(cluster_pick.split()[-1])
        text_blob = " ".join(df[df["kmeans_cluster"] == cluster_id_pick]["text_preprocessed"])
        if text_blob.strip():
            wc2 = WordCloud(width=1100, height=380, background_color="white", mode="RGB",
                             colormap="RdPu", max_words=80).generate(text_blob)
            fig_wc2, ax_wc2 = plt.subplots(figsize=(12, 4))
            fig_wc2.patch.set_alpha(0)
            ax_wc2.imshow(wc2, interpolation="bilinear")
            ax_wc2.axis("off")
            st.pyplot(fig_wc2)
        else:
            st.info("Tidak cukup data untuk WordCloud.")

        st.subheader("Crosstab: Klaster K-Means vs Label Urgensi")
        st.caption("Bukan confusion matrix klasifikasi sebenarnya (K-Means unsupervised) — ini menunjukkan sejauh mana klaster yang terbentuk selaras dengan label urgensi leksikon.")
        cross = pd.crosstab(df["kmeans_cluster"], df["urgensi"])
        fig_cross = px.imshow(cross, text_auto=True, color_continuous_scale=PASTEL_SCALE,
                               labels=dict(x="Urgensi", y="Klaster K-Means", color="Jumlah"))
        fig_cross.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_cross, use_container_width=True)

        insight_card("Insight Otomatis — K-Means", generate_kmeans_insight(df, sil, n_clusters))

    # -------- TAB 5: ML CLASSIFIER --------
    with tab5:
        st.subheader("🤖 ML Classifier — TF-IDF + Logistic Regression")
        st.caption("Model dilatih untuk memprediksi label **urgensi** (hasil leksikon) dari teks, sebagai pembanding terhadap pendekatan rule-based.")

        labels_list = df["urgensi"].tolist()
        class_counts = Counter(labels_list)
        if len(class_counts) < 2 or min(class_counts.values()) < 2:
            st.warning("⚠️ Data terlalu sedikit / tidak seimbang antar kelas urgensi untuk melatih classifier yang andal. Tambahkan lebih banyak data agar evaluasi lebih bermakna.")
        else:
            model, ml_vectorizer, ml_classes, report, cm, acc, f1m, n_test = train_ml_classifier(
                df["text_preprocessed"].tolist(), labels_list
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Akurasi (test set)", f"{acc*100:.1f}%")
            c2.metric("F1-Score (macro)", f"{f1m:.3f}")
            c3.metric("Jumlah Data Uji", n_test)

            st.markdown("#### Classification Report")
            report_df = pd.DataFrame(report).T
            report_df = report_df.loc[[c for c in ml_classes if c in report_df.index] + ["accuracy", "macro avg", "weighted avg"]]
            report_df = report_df.round(3)
            st.dataframe(report_df, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### F1-Score per Kelas")
                f1_per_class = pd.DataFrame({
                    "Kelas": ml_classes,
                    "F1-Score": [report[c]["f1-score"] for c in ml_classes],
                })
                fig_f1 = px.bar(f1_per_class, x="Kelas", y="F1-Score", color="Kelas",
                                 color_discrete_map=COLOR_MAP_URGENCY, range_y=[0, 1], text="F1-Score")
                fig_f1.update_traces(texttemplate="%{text:.2f}")
                fig_f1.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                st.plotly_chart(fig_f1, use_container_width=True)
            with c2:
                st.markdown("#### Confusion Matrix")
                fig_cm = px.imshow(cm, x=ml_classes, y=ml_classes, text_auto=True,
                                    color_continuous_scale=PASTEL_SCALE,
                                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"))
                fig_cm.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True)

            insight_card("Insight Otomatis — ML Classifier", [
                f"Model mencapai akurasi <b>{acc*100:.1f}%</b> dan F1-score makro <b>{f1m:.3f}</b> pada data uji.",
                f"Kelas dengan F1-score tertinggi: <b>{max(ml_classes, key=lambda c: report[c]['f1-score'])}</b>.",
                "Karena label berasal dari pendekatan leksikon (bukan anotasi manusia), classifier ini paling berguna sebagai pembanding kecepatan/konsistensi terhadap aturan leksikon, bukan sebagai ground truth absolut.",
            ])

            st.divider()
            st.markdown("#### 🔮 Cek Curhatan Baru dengan Model")
            new_text = st.text_area("Masukkan teks curhatan:", height=110,
                                     placeholder="Contoh: Aku capek banget sama tugas kuliah yang numpuk terus...")
            if st.button("🔍 Analisis Sekarang", type="primary") and new_text.strip():
                low = new_text.lower()
                label, score, sev, mod, mild_ = score_urgency(low)
                klaster = detect_cluster(low)
                supportive = detect_support(low)
                preprocessed = preprocess(new_text)

                X_new_bow = bow_vec.transform([preprocessed])
                doc_top_new = lda_model.transform(X_new_bow)
                lda_pred = doc_top_new.argmax()

                X_new_tfidf = ml_vectorizer.transform([preprocessed])
                ml_pred = model.predict(X_new_tfidf)[0]
                ml_proba = dict(zip(model.classes_, model.predict_proba(X_new_tfidf)[0]))

                colr1, colr2 = st.columns(2)
                with colr1:
                    if "Darurat" in label:
                        st.error(f"### Leksikon: {label}")
                    elif "Perhatian" in label:
                        st.warning(f"### Leksikon: {label}")
                    else:
                        st.success(f"### Leksikon: {label}")

                    st.markdown(f"""
                    <div class="mw-card">
                    <h4>Hasil Analisis</h4>
                    <p class="mw-mono-val">
                    🎯 <b>Skor Urgensi (leksikon):</b> {score:.1f}<br>
                    🤖 <b>Prediksi ML Classifier:</b> {ml_pred}<br>
                    🧩 <b>Klaster Akar Masalah:</b> {klaster}<br>
                    🔬 <b>Topik LDA Dominan:</b> Topik {lda_pred} ({', '.join(topic_words[lda_pred][:3])})<br>
                    💗 <b>Terdeteksi Supportif:</b> {"Ya ✅" if supportive else "Tidak"}
                    </p>
                    </div>
                    """, unsafe_allow_html=True)

                with colr2:
                    st.markdown("**Probabilitas Prediksi Model:**")
                    proba_df = pd.DataFrame({"Kelas": list(ml_proba.keys()), "Probabilitas": list(ml_proba.values())})
                    fig_proba = px.bar(proba_df, x="Kelas", y="Probabilitas", color="Kelas",
                                        color_discrete_map=COLOR_MAP_URGENCY, range_y=[0, 1])
                    fig_proba.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                    st.plotly_chart(fig_proba, use_container_width=True)
                    st.markdown("**Teks Setelah Preprocessing:**")
                    st.code(preprocessed if preprocessed else "(teks kosong setelah preprocessing)")
                    if "Darurat" in label:
                        st.error(f"**🆘 Hubungi layanan darurat:**\n{CRISIS_RESOURCES}")

    # -------- TAB 6: SNA --------
    with tab6:
        st.subheader("🕸️ Social Network Analysis — Peta Support System")
        st.caption("Warna edge **pink** = interaksi biasa · **mint** = respon supportif")

        if G.number_of_nodes() == 0:
            st.warning("Tidak ada relasi (@mention / reply) terdeteksi. Pastikan kolom reply-to / teks mengandung @mention.")
        else:
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
            g2.metric("Edge (Interaksi)", G.number_of_edges())
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
                for acc_, val in top5_deg:
                    st.markdown(f"<span class='mw-mono-val'>@{acc_}: {val:.3f}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with cb:
                st.markdown("<div class='mw-card'><h4>Betweenness Centrality</h4>", unsafe_allow_html=True)
                for acc_, val in top5_btw:
                    st.markdown(f"<span class='mw-mono-val'>@{acc_}: {val:.3f}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with cc:
                st.markdown("<div class='mw-card'><h4>Eigenvector Centrality</h4>", unsafe_allow_html=True)
                if top5_eig:
                    for acc_, val in top5_eig:
                        st.markdown(f"<span class='mw-mono-val'>@{acc_}: {val:.3f}</span>", unsafe_allow_html=True)
                else:
                    st.caption("Tidak dapat dihitung (graf tidak konvergen)")
                st.markdown("</div>", unsafe_allow_html=True)

            insight_card("Insight Otomatis — SNA", generate_sna_insight(G))

    # -------- TAB 7: DATA --------
    with tab7:
        st.subheader("📋 Tabel Data Lengkap")
        f1, f2 = st.columns(2)
        with f1:
            urg_filter = st.multiselect("Filter Urgensi:", df["urgensi"].unique(), default=list(df["urgensi"].unique()))
        with f2:
            klaster_filter = st.multiselect("Filter Klaster:", df["klaster_leksikon"].unique(),
                                             default=list(df["klaster_leksikon"].unique()))

        filtered = df[df["urgensi"].isin(urg_filter) & df["klaster_leksikon"].isin(klaster_filter)]
        display_cols = [text_col, "urgensi", "klaster_leksikon", "lda_topic", "kmeans_cluster",
                         "skor_urgensi", "is_support", "jumlah_kata"]
        if username_col:
            display_cols.insert(0, username_col)
        st.dataframe(filtered[[c for c in display_cols if c in filtered.columns]],
                     use_container_width=True, height=400)

        csv_out = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Hasil Analisis (CSV)", csv_out,
                            "mindwatch_hasil.csv", "text/csv")

    st.markdown(f"""
    <div class="mw-footer">
    🌷 <b>MindWatch Dashboard</b> — Riset & Kesadaran Sosial Isu Kesehatan Mental Gen Z<br>
    {CRISIS_RESOURCES}
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="mw-card" style="text-align:center; padding:40px;">
    <h4>👈 Mulai dengan Upload Dataset CSV</h4>
    <p style="font-size:14.5px; max-width:600px; margin: 12px auto;">
    Upload file CSV hasil scraping Twitter/X atau platform media sosial lainnya.<br>
    Disarankan memiliki kolom: <b>full_text</b>, <b>username</b>, <b>in_reply_to_screen_name</b>, <b>created_at</b>
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ Fitur Dashboard")
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🧪", "Preprocessing Pipeline", "Tahapan lengkap dari raw text hingga Sastrawi stemming, dengan contoh nyata & statistik kata."),
        ("📐", "Statistik Deskriptif", "Panjang teks, distribusi jam/tanggal posting, top words, dan engagement."),
        ("🚦", "Urgensi & Insight", "Klasifikasi Darurat · Perlu Perhatian · Ringan dengan insight otomatis."),
        ("🤖", "LDA · K-Means · ML", "Topic modeling, clustering, dan classifier dengan classification report & confusion matrix."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="mw-card">
            <h4>{icon} {title}</h4>
            <p style="font-size:13px;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="mw-footer">{CRISIS_RESOURCES}</div>
    """, unsafe_allow_html=True)
