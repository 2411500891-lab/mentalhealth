import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import networkx as nx

# ================= KONFIGURASI =================

st.set_page_config(
    page_title="Mental Health Gen Z Dashboard",
    layout="wide"
)

# ================= CSS AESTHETIC =================

st.markdown("""
<style>

.stApp{
background-color:#0E1117;
color:white;
}

h1{
color:#5DADE2;
text-align:center;
}

[data-testid="stMetric"]{
background-color:#1E1E2F;
padding:15px;
border-radius:15px;
}

</style>
""",unsafe_allow_html=True)

# ================= PREPROCESSING =================

STOPWORDS=["yang","dan","di","ke","dari","itu"]

def preprocess(text):

    text=str(text).lower()

    text=re.sub(r'[^a-zA-Z ]','',text)

    words=text.split()

    words=[x for x in words if x not in STOPWORDS]

    return " ".join(words)

# ================= UI =================

st.title("🧠 Mental Health Gen Z Monitoring Dashboard")

with st.sidebar:

    st.header("Upload Dataset")

    uploaded_file=st.file_uploader(
        "Upload CSV Twitter/X",
        type=["csv"]
    )

if uploaded_file:

    df=pd.read_csv(uploaded_file)

    st.success(
        f"Dataset berhasil dimuat: {len(df)} data"
    )

    text_col=st.selectbox(
        "Pilih kolom tweet:",
        df.columns
    )

    df["clean_text"]=df[text_col].apply(preprocess)

#=================== KLASIFIKASI ====================

    def urgency(text):

        urgent_keywords=[

        "depresi",
        "capek hidup",
        "ingin menyerah",
        "sendiri"

        ]

        for word in urgent_keywords:

            if word in text:

                return "Butuh Pertolongan Segera"

        return "Curhat Ringan"

    df["urgensi"]=df["clean_text"].apply(
        urgency
    )

#=================== KPI ====================

    urgent=(
        df["urgensi"]=="Butuh Pertolongan Segera"
    ).sum()

    total=len(df)

    persen=urgent/total*100

    c1,c2,c3=st.columns(3)

    c1.metric(
        "Total Tweet",
        total
    )

    c2.metric(
        "Darurat",
        urgent
    )

    c3.metric(
        "Persentase",
        f"{persen:.2f}%"
    )

#=================== LAMPU INDIKATOR ====================

    st.subheader(
        "🚨 Lampu Indikator Urgensi"
    )

    if persen>40:

        st.error(
        "🔴 DARURAT"
        )

    elif persen>20:

        st.warning(
        "🟡 PERLU PERHATIAN"
        )

    else:

        st.success(
        "🟢 AMAN"
        )

#=================== KLASTER ====================

    st.subheader(
        "📊 Klaster Penyebab"
    )

    vectorizer=TfidfVectorizer()

    X=vectorizer.fit_transform(
        df["clean_text"]
    )

    kmeans=KMeans(
        n_clusters=3,
        random_state=42
    )

    df["cluster"]=kmeans.fit_predict(X)

    cluster_map={

    0:"Tekanan Akademik",
    1:"Masalah Keluarga",
    2:"Finansial"

    }

    df["cluster"]=df[
    "cluster"
    ].map(cluster_map)

    fig,ax=plt.subplots()

    df["cluster"].value_counts().plot.pie(
        autopct="%1.1f%%",
        ax=ax
    )

    st.pyplot(fig)

#=================== WORDCLOUD ====================

    st.subheader(
    "☁️ Trending Topic"
    )

    text=" ".join(
    df["clean_text"]
    )

    wc=WordCloud(
        width=1000,
        height=500
    ).generate(text)

    fig,ax=plt.subplots(
        figsize=(10,5)
    )

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig)

#=================== SNA ====================

    st.subheader(
    "🌐 Support System Network"
    )

    G=nx.Graph()

    G.add_edge(
        "akunA",
        "akunB"
    )

    G.add_edge(
        "akunB",
        "akunC"
    )

    fig,ax=plt.subplots()

    nx.draw(
        G,
        with_labels=True,
        ax=ax
    )

    st.pyplot(fig)

#=================== DATA ====================

    st.subheader(
    "Data Hasil"
    )

    st.dataframe(df)

else:

    st.info(
    "Upload CSV Twitter/X untuk memulai"
    )
