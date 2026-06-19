import re
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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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
.mw-light.on-red    { background: #ef4444; opacity: 1; box-shadow: 0 0 35px 8px rgba(239,68,68,0.75); animation: pulse 1.4s infinite; }
.mw-light.on-yellow { background: #f59e0b; opacity: 1; box-shadow: 0 0 35px 8px rgba(245,158,11,0.6); }
.mw-light.on-green  { background: #10b981; opacity: 1; box-shadow: 0 0 30px 6px rgba(16,185,129,0.55); }
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

/* Divider */
hr { border-color: rgba(255,105,180,0.15) !important; }

/* DataFrame */
[data-testid="stDataFrame"] { border: 1px solid rgba(255,105,180,0.18) !important; border-radius: 12px !important; }

/* Footer */
.mw-footer {
    margin-top: 30px; padding: 18px; border-radius: 16px;
    background: rgba(255,20,100,0.06); border: 1px solid rgba(255,105,180,0.15);
    font-size: 12.5px; color: #d09ab8; text-align: center;
}

/* Badge */
.mw-badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px; }
.badge-urgent { background:rgba(239,68,68,0.18); color:#fca5a5; border:1px solid rgba(239,68,68,0.4); }
.badge-watch  { background:rgba(245,158,11,0.18); color:#fcd34d; border:1px solid rgba(245,158,11,0.4); }
.badge-mild   { background:rgba(255,105,180,0.18); color:#ffb6d9; border:1px solid rgba(255,105,180,0.4); }
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

TOPIC_LABELS = {
    0: "Edukasi & Manajemen Diri",
    1: "Klinis (Depresi & Psikolog)",
    2: "Gangguan Kecemasan (Anxiety)",
}

# ============================================================
# PREPROCESSING
# ============================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_text(text):
    words = text.split()
    return " ".join(NORMALIZATION_DICT.get(w, w) for w in words)

def remove_stopwords(text):
    words = text.split()
    return " ".join(w for w in words if (w not in STOPWORDS or w in KEEP_WORDS) and len(w) > 2)

def preprocess(text):
    t = clean_text(text)
    t = normalize_text(t)
    t = remove_stopwords(t)
    return t

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

    # Elbow
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
# PLOTLY THEME HELPER
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f5e6f0",
    font_family="Nunito",
)

PINK_COLORS = ["#ff69b4", "#ff1493", "#ffb6d9", "#c0507a", "#e0709a", "#ff9ec8"]
COLOR_MAP_URGENCY = {
    "🚨 Darurat": "#ef4444",
    "⚠️ Perlu Perhatian": "#f59e0b",
    "💬 Curhat Ringan": "#ff69b4",
}

# ============================================================
# UI — HERO
# ============================================================
st.markdown("""
<div class="mw-hero">
  <h1>🧠 MindWatch — Monitoring Isu Kesehatan Mental Gen Z</h1>
  <p>Deteksi urgensi · LDA Topic Modeling · K-Means Clustering · Social Network Analysis (SNA)</p>
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
    st.markdown("### ℹ️ Tentang")
    st.caption("**Tema 7** — Monitoring Isu Kesehatan Mental Gen Z\nKomponen: Klasifikasi Urgensi · LDA · K-Means · SNA")
    st.divider()
    st.markdown("### 🆘 Krisis? Hubungi:")
    st.caption(CRISIS_RESOURCES)

# ============================================================
# MAIN CONTENT
# ============================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset dimuat — **{len(df):,} baris**, **{len(df.columns)} kolom**")

    # ---- Konfigurasi kolom ----
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

    # ---- Preprocessing & klasifikasi ----
    with st.spinner("🌸 Memproses data..."):
        df["text_clean"] = df[text_col].astype(str).str.lower()
        df["text_preprocessed"] = df[text_col].astype(str).apply(preprocess)

        urgency_results = df["text_clean"].apply(score_urgency)
        df["urgensi"]       = urgency_results.apply(lambda x: x[0])
        df["skor_urgensi"]  = urgency_results.apply(lambda x: x[1])
        df["n_severe"]      = urgency_results.apply(lambda x: x[2])
        df["n_moderate"]    = urgency_results.apply(lambda x: x[3])
        df["n_mild"]        = urgency_results.apply(lambda x: x[4])
        df["klaster_leksikon"] = df["text_clean"].apply(detect_cluster)
        df["is_support"]    = df["text_clean"].apply(detect_support)

        if ts_col:
            df["_ts_parsed"] = pd.to_datetime(df[ts_col], errors="coerce")

        # LDA
        clean_texts = df["text_preprocessed"].tolist()
        dom_topics, topic_words, log_lik, perplexity, lda_model, bow_vec = run_lda(clean_texts, n_topics)
        df["lda_topic"] = dom_topics
        topic_label_map = {i: f"Topik {i}: {', '.join(topic_words[i][:3])}" for i in range(n_topics)}
        df["lda_topic_label"] = df["lda_topic"].map(topic_label_map)

        # K-Means
        km_labels, X_2d, km_final, k_range, wcss, sil, dbi, chi = run_kmeans(clean_texts, n_clusters)
        df["kmeans_cluster"] = km_labels

        # SNA
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

    # ---- Indikator status ----
    st.markdown("### 🚦 Indikator Urgensi")
    col_light, col_kpi = st.columns([1, 2.5])

    with col_light:
        st.markdown(f"""
        <div class="mw-light-wrap">
          <div class="mw-light" style="background:#ef4444; opacity:{1 if light_class=='on-red' else 0.15}; {'box-shadow:0 0 35px 8px rgba(239,68,68,0.75);animation:pulse 1.4s infinite;' if light_class=='on-red' else ''}"></div>
          <div class="mw-light" style="background:#f59e0b; opacity:{1 if light_class=='on-yellow' else 0.15}; {'box-shadow:0 0 35px 8px rgba(245,158,11,0.6);' if light_class=='on-yellow' else ''}"></div>
          <div class="mw-light" style="background:#10b981; opacity:{1 if light_class=='on-green' else 0.15}; {'box-shadow:0 0 30px 6px rgba(16,185,129,0.55);' if light_class=='on-green' else ''}"></div>
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Urgensi & Tren",
        "🔬 LDA Topic Modeling",
        "🧩 K-Means Clustering",
        "🕸️ Social Network (SNA)",
        "📋 Data & Export",
        "🔮 Cek Curhatan Baru",
    ])

    # -------- TAB 1: URGENSI --------
    with tab1:
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
                             color_discrete_sequence=PINK_COLORS)
            fig_kp.update_layout(**PLOTLY_LAYOUT, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_kp, use_container_width=True)
        with c4:
            fig_kb = px.bar(klaster_count.sort_values("Jumlah"), x="Jumlah", y="Klaster",
                             orientation="h", color="Jumlah",
                             color_continuous_scale=[[0,"#2d1022"],[1,"#ff69b4"]])
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

    # -------- TAB 2: LDA --------
    with tab2:
        st.subheader("🔬 Latent Dirichlet Allocation (LDA) Topic Modeling")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="mw-card">
            <h4>Metrik Evaluasi Model LDA</h4>
            <p style="color:#f0c0d8; font-size:14px;">
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
            <p style="color:#f0c0d8; font-size:13px;">
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
                <p style="color:#ffb6d9; font-size:13px; line-height:1.8;">{words_str}</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("Distribusi Dokumen per Topik")
        topic_dist = df["lda_topic"].value_counts().reset_index()
        topic_dist.columns = ["Topik", "Jumlah"]
        topic_dist["Label"] = topic_dist["Topik"].map(lambda x: f"Topik {x}: {', '.join(topic_words[x][:2])}")
        fig_lda = px.bar(topic_dist, x="Label", y="Jumlah",
                          color="Jumlah", color_continuous_scale=[[0,"#2d1022"],[1,"#ff69b4"]],
                          text="Jumlah")
        fig_lda.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, xaxis_title="", yaxis_title="Jumlah Dokumen")
        st.plotly_chart(fig_lda, use_container_width=True)

        st.subheader("Tren Topik LDA dari Waktu ke Waktu")
        if ts_col and df["_ts_parsed"].notna().sum() > 0:
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
                    line=dict(color=PINK_COLORS[i % len(PINK_COLORS)], width=2.5),
                ))
            fig_lda_trend.update_layout(**PLOTLY_LAYOUT, xaxis_title="Tanggal", yaxis_title="Volume Isu",
                                         title="Deteksi Tren Isu Kesehatan Mental (LDA Line Chart)")
            st.plotly_chart(fig_lda_trend, use_container_width=True)
        else:
            st.info("💡 Pilih kolom waktu di konfigurasi untuk melihat tren topik harian.")

    # -------- TAB 3: K-MEANS --------
    with tab3:
        st.subheader("🧩 K-Means Clustering")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="mw-card">
            <h4>Metrik Evaluasi K-Means (K={n_clusters})</h4>
            <p style="color:#f0c0d8; font-size:14px;">
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
            <p style="color:#f0c0d8; font-size:13px;">
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
                                            line=dict(color="#ff69b4", width=2.5),
                                            marker=dict(size=8, color="#ff1493")))
            fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="#ffb6d9",
                                 annotation_text=f"K={n_clusters}", annotation_font_color="#ffb6d9")
            fig_elbow.update_layout(**PLOTLY_LAYOUT, xaxis_title="Jumlah Klaster (K)", yaxis_title="WCSS (Inertia)")
            st.plotly_chart(fig_elbow, use_container_width=True)

        with c2:
            st.subheader("Visualisasi Klaster (2D)")
            df_plot = pd.DataFrame({"x": X_2d[:, 0], "y": X_2d[:, 1], "Klaster": df["kmeans_cluster"].astype(str)})
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
                <p style="color:#e0c0d0; font-size:12px;">📄 {n_docs} dokumen</p>
                <p style="color:#ffb6d9; font-size:13px; line-height:1.8;">{words_str}</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("WordCloud per Klaster")
        cluster_pick = st.selectbox("Pilih Klaster:", [f"Klaster {i}" for i in range(n_clusters)])
        cluster_id_pick = int(cluster_pick.split()[-1])
        text_blob = " ".join(df[df["kmeans_cluster"] == cluster_id_pick]["text_preprocessed"])
        if text_blob.strip():
            wc = WordCloud(width=1100, height=380, background_color=None, mode="RGBA",
                            colormap="RdPu", max_words=80).generate(text_blob)
            fig_wc, ax_wc = plt.subplots(figsize=(12, 4))
            fig_wc.patch.set_alpha(0)
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("Tidak cukup data untuk WordCloud.")

    # -------- TAB 4: SNA --------
    with tab4:
        st.subheader("🕸️ Social Network Analysis — Peta Support System")
        st.caption("Warna edge **pink** = interaksi biasa · **pink terang** = respon supportif")

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

            # Centrality metrics
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

    # -------- TAB 5: DATA --------
    with tab5:
        st.subheader("📋 Tabel Data Lengkap")
        f1, f2 = st.columns(2)
        with f1:
            urg_filter = st.multiselect("Filter Urgensi:", df["urgensi"].unique(), default=list(df["urgensi"].unique()))
        with f2:
            klaster_filter = st.multiselect("Filter Klaster:", df["klaster_leksikon"].unique(),
                                             default=list(df["klaster_leksikon"].unique()))

        filtered = df[df["urgensi"].isin(urg_filter) & df["klaster_leksikon"].isin(klaster_filter)]
        display_cols = [text_col, "urgensi", "klaster_leksikon", "lda_topic", "kmeans_cluster", "skor_urgensi", "is_support"]
        if username_col:
            display_cols.insert(0, username_col)
        st.dataframe(filtered[[c for c in display_cols if c in filtered.columns]],
                     use_container_width=True, height=400)

        csv_out = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Hasil Analisis (CSV)", csv_out,
                            "mindwatch_hasil.csv", "text/csv")

    # -------- TAB 6: CEK BARU --------
    with tab6:
        st.subheader("🔮 Cek Tingkat Urgensi Curhatan Baru")
        new_text = st.text_area("Masukkan teks curhatan:", height=110,
                                 placeholder="Contoh: Aku capek banget sama tugas kuliah yang numpuk terus...")
        if st.button("🔍 Analisis Sekarang", type="primary") and new_text.strip():
            low = new_text.lower()
            label, score, sev, mod, mild_ = score_urgency(low)
            klaster = detect_cluster(low)
            supportive = detect_support(low)
            preprocessed = preprocess(new_text)

            # LDA prediction
            X_new_bow = bow_vec.transform([preprocessed])
            doc_top_new = lda_model.transform(X_new_bow)
            lda_pred = doc_top_new.argmax()

            colr1, colr2 = st.columns(2)
            with colr1:
                if "Darurat" in label:
                    st.error(f"### {label}")
                elif "Perhatian" in label:
                    st.warning(f"### {label}")
                else:
                    st.success(f"### {label}")

                st.markdown(f"""
                <div class="mw-card">
                <h4>Hasil Analisis</h4>
                <p style="color:#f0c0d8; font-size:14px;">
                🎯 <b>Skor Urgensi:</b> {score:.1f}<br>
                🧩 <b>Klaster Akar Masalah:</b> {klaster}<br>
                🔬 <b>Topik LDA Dominan:</b> Topik {lda_pred} ({', '.join(topic_words[lda_pred][:3])})<br>
                💗 <b>Terdeteksi Supportif:</b> {"Ya ✅" if supportive else "Tidak"}
                </p>
                </div>
                """, unsafe_allow_html=True)

            with colr2:
                st.markdown("**Teks Setelah Preprocessing:**")
                st.code(preprocessed if preprocessed else "(teks kosong setelah preprocessing)")
                if "Darurat" in label:
                    st.error(f"**🆘 Hubungi layanan darurat:**\n{CRISIS_RESOURCES}")

    # ---- Footer ----
    st.markdown(f"""
    <div class="mw-footer">
    🌸 <b>MindWatch Dashboard</b> — Riset & Kesadaran Sosial Isu Kesehatan Mental Gen Z<br>
    {CRISIS_RESOURCES}
    </div>
    """, unsafe_allow_html=True)

else:
    # Landing page
    st.markdown("""
    <div class="mw-card" style="text-align:center; padding:40px;">
    <h4>👈 Mulai dengan Upload Dataset CSV</h4>
    <p style="color:#f0c0d8; font-size:14.5px; max-width:600px; margin: 12px auto;">
    Upload file CSV hasil scraping Twitter/X atau platform media sosial lainnya.<br>
    Disarankan memiliki kolom: <b>full_text</b>, <b>username</b>, <b>in_reply_to_screen_name</b>, <b>created_at</b>
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ Fitur Dashboard")
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🚦", "Indikator Urgensi", "Klasifikasi otomatis: Darurat · Perlu Perhatian · Ringan dengan lampu status real-time."),
        ("🔬", "LDA Topic Modeling", "Pemodelan topik tersembunyi menggunakan Latent Dirichlet Allocation + Bag of Words."),
        ("🧩", "K-Means Clustering", "Klasterisasi teks dengan TF-IDF + SVD + K-Means dan evaluasi Silhouette/DBI/CH."),
        ("🕸️", "Social Network (SNA)", "Visualisasi jejaring mention + ranking centrality: Degree, Betweenness, Eigenvector."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="mw-card">
            <h4>{icon} {title}</h4>
            <p style="color:#f0c0d8; font-size:13px;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="mw-footer">{CRISIS_RESOURCES}</div>
    """, unsafe_allow_html=True)
