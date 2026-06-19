import streamlit as st

st.set_page_config(
    page_title="Mental Health Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Mental Health Gen Z Dashboard")
st.write("Tes tampil")

"""
data_generator.py
==================
Modul untuk men-generate dataset DUMMY yang merepresentasikan tweet/menfess
bertema kesehatan mental Gen Z (mensimulasikan hasil scraping dari akun
menfess seperti @schfess dengan keyword: burnout kuliah, depresi, anxiety,
butuh teman cerita, mental health).

Dataset ini dibuat deterministik (seed tetap) agar dashboard menampilkan
hasil yang konsisten setiap kali dijalankan, sekaligus tetap terlihat
realistis untuk keperluan presentasi tugas kampus.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -----------------------------------------------------------------------
# 1. BANK KALIMAT TWEET PER KATEGORI URGENSI
# -----------------------------------------------------------------------
# Kategori "Butuh Pertolongan Segera" -> kalimat dengan indikasi kuat
# distress berat (capek hidup, gak kuat lagi, dsb)
URGENT_TEMPLATES = [
    "udah gak kuat lagi jujur, capek banget sama hidup ini #burnoutkuliah",
    "tiap malam nangis sendirian, rasanya berat banget mau cerita ke siapa #depresi",
    "anxiety gue parah banget minggu ini, susah napas tiap mau ke kampus",
    "butuh teman cerita beneran, udah gak tau harus cerita ke siapa lagi",
    "rasanya kayak gak ada semangat sama sekali buat ngelanjutin semua ini #mentalhealth",
    "skripsi numpuk, keluarga gak ngerti, aku bener2 di titik terendah",
    "udah 3 hari susah makan dan susah tidur, rasanya capek banget",
    "udah cape banget, ngerasa jadi beban buat orang-orang sekitar",
    "burnout kuliah separah ini baru kerasa, tiap hari nangis tanpa sebab",
    "anxiety attack lagi di tengah kelas, gemeteran sampe gabisa fokus",
    "depresi ini udah ganggu banget, tidur gabisa, makan males, semua hampa",
    "gak ada yang notice kalau aku struggling, butuh pertolongan beneran",
    "akhir-akhir ini susah banget ngerasain hal yang biasanya bikin senang",
    "udah cape pura-pura baik-baik aja di depan temen-temen",
    "tiap malam mikirin masalah ini terus, susah banget buat tenang",
]

# Kategori "Curhat Ringan" -> keluhan sehari-hari, masih dalam taraf wajar
LIGHT_TEMPLATES = [
    "capek banget tugas numpuk minggu ini tapi yaudah dijalanin aja",
    "burnout kuliah dikit nih abis ujian, butuh healing weekend ini",
    "anxiety sebelum presentasi tapi untungnya lancar tadi",
    "butuh teman cerita buat planning liburan abis UAS, ada yang mau ikut?",
    "mental health itu penting, jangan lupa istirahat ya guys",
    "stress dikit mikirin deadline tapi masih semangat kok",
    "kadang ngerasa burnout kuliah online gini, kangen kelas offline",
    "self reminder: gapapa kalau hari ini belum produktif banget",
    "lagi belajar coping stress yang lebih sehat, recommend dong caranya",
    "deg-degan nunggu nilai keluar, semoga hasilnya baik",
    "ngerasa lelah aja sama rutinitas kuliah, butuh me time",
    "curhat dikit, dosen pembimbing susah dihubungin huhu",
    "mental health check-in: hari ini lumayan baik kok alhamdulillah",
    "anxiety dikit jelang sidang tapi udah persiapan matang",
    "butuh teman cerita yang santai aja buat ngobrolin drama kuliah",
    "capek sama circle pertemanan yang toxic, tapi masih bisa diatasi",
    "burnout kuliah parah pas masa krs-an, untung udah kelar",
    "self healing time, nonton drakor abis ngerjain laporan praktikum",
]

# -----------------------------------------------------------------------
# 2. PENYEBAB / KLASTER ISU (dipakai utk topic modeling sederhana)
# -----------------------------------------------------------------------
CLUSTER_INFO = {
    "Tekanan Akademik": {
        "keywords": ["skripsi", "deadline", "ujian", "dosen", "tugas", "kuliah", "sidang", "krs", "ipk"],
        "share": 0.42,
    },
    "Masalah Keluarga": {
        "keywords": ["ortu", "keluarga", "rumah", "ayah", "ibu", "broken home", "ekspektasi keluarga"],
        "share": 0.27,
    },
    "Finansial": {
        "keywords": ["uang", "ukt", "biaya kuliah", "kerja part time", "beasiswa", "finansial", "bayar kos"],
        "share": 0.31,
    },
}

CLUSTER_TWEET_TEMPLATES = {
    "Tekanan Akademik": [
        "deadline tugas numpuk barengan sidang skripsi, kepala mau pecah",
        "dosen pembimbing skripsi susah ditemui, makin stress mikirin sidang",
        "ipk turun gara-gara burnout kuliah semester ini, takut gak lulus tepat waktu",
        "ujian besok 3 mata kuliah, anxiety mikirin gak sempet belajar",
        "revisi skripsi udah ke-7 kali, rasanya capek banget sama akademik",
    ],
    "Masalah Keluarga": [
        "ekspektasi keluarga ketinggian, capek selalu dibanding-bandingin sama saudara",
        "broken home bikin susah fokus kuliah, mental health makin terganggu",
        "ortu gak ngerti perjuangan kuliah, tiap pulang malah berantem",
        "rumah harusnya jadi tempat ternyaman tapi malah jadi sumber stress",
        "konflik keluarga bikin anxiety tiap mau pulang ke rumah",
    ],
    "Finansial": [
        "UKT belum kebayar, mikirin biaya kuliah bikin susah tidur tiap malam",
        "kerja part time sambil kuliah biar bisa bayar kos, capek tapi harus kuat",
        "finansial keluarga lagi susah, takut gak bisa lanjut semester depan",
        "beasiswa dicabut gara-gara ipk turun, makin pusing mikirin biaya",
        "uang bulanan telat, makan seadanya sambil mikirin tugas kuliah",
    ],
}

# -----------------------------------------------------------------------
# 3. NAMA AKUN (anonim/fiktif) UNTUK SUPPORT SYSTEM NETWORK
# -----------------------------------------------------------------------
SUPPORTER_HANDLES = [
    "@kak_pendengar", "@temancurhat_id", "@psikologmuda", "@hangatpelukan",
    "@sahabatdengar", "@mentalsupport_id", "@kawanresah", "@pelukvirtual",
    "@dengerinaja", "@temenceritamu", "@ruangpulih", "@hellopeduli",
]

CURHATOR_HANDLES = [f"@curhatanon{idx:03d}" for idx in range(1, 61)]

SUPPORT_REPLY_TEMPLATES = [
    "semangat ya kak, kamu gak sendirian kok 🤍",
    "boleh cerita lebih lanjut, aku dengerin kok",
    "kamu udah hebat banget bertahan sejauh ini",
    "coba hubungi layanan konseling kampus ya, mereka siap bantu",
    "pelukan jarak jauh, semoga lekas membaik ya",
    "it's okay to not be okay, take your time",
    "kalau butuh teman cerita, DM aku aja ya",
    "kamu berharga, jangan nyerah ya kak",
]


def _random_dates(n, days_back=30):
    """Generate tanggal random dalam rentang 30 hari terakhir."""
    base = datetime(2026, 6, 19)
    offsets = np.random.randint(0, days_back, size=n)
    seconds = np.random.randint(0, 86400, size=n)
    return [base - timedelta(days=int(o), seconds=int(s)) for o, s in zip(offsets, seconds)]


def generate_tweets(n_total=620):
    """
    Generate dataframe tweet dummy lengkap dengan:
    - text, tanggal, keyword pemicu
    - label urgensi (Butuh Pertolongan Segera / Curhat Ringan)
    - label klaster penyebab (Tekanan Akademik / Masalah Keluarga / Finansial)
    - jumlah likes & replies (untuk simulasi engagement)
    """
    rows = []

    # Proporsi urgensi: dibuat agak tinggi (~33%) supaya gauge & alert
    # terlihat dinamis saat presentasi, namun tetap di bawah ambang 40%
    # secara default (dosen bisa lihat behavior alert via slider simulasi).
    n_urgent = int(n_total * 0.33)
    n_light = n_total - n_urgent

    keyword_pool = ["Burnout Kuliah", "Depresi", "Anxiety", "Butuh Teman Cerita", "Mental Health"]

    # --- generate kategori urgent ---
    for i in range(n_urgent):
        text = np.random.choice(URGENT_TEMPLATES)
        cluster = np.random.choice(list(CLUSTER_INFO.keys()),
                                    p=[CLUSTER_INFO[c]["share"] for c in CLUSTER_INFO])
        rows.append({
            "tweet_id": f"T{len(rows)+1:04d}",
            "akun": np.random.choice(CURHATOR_HANDLES),
            "text": text,
            "keyword": np.random.choice(keyword_pool),
            "label_urgensi": "Butuh Pertolongan Segera",
            "klaster_penyebab": cluster,
            "likes": np.random.randint(5, 80),
            "replies": np.random.randint(0, 25),
        })

    # --- generate kategori ringan ---
    for i in range(n_light):
        text = np.random.choice(LIGHT_TEMPLATES)
        cluster = np.random.choice(list(CLUSTER_INFO.keys()),
                                    p=[CLUSTER_INFO[c]["share"] for c in CLUSTER_INFO])
        rows.append({
            "tweet_id": f"T{len(rows)+1:04d}",
            "akun": np.random.choice(CURHATOR_HANDLES),
            "text": text,
            "keyword": np.random.choice(keyword_pool),
            "label_urgensi": "Curhat Ringan",
            "klaster_penyebab": cluster,
            "likes": np.random.randint(10, 150),
            "replies": np.random.randint(0, 40),
        })

    # --- selipkan beberapa tweet eksplisit per klaster (biar wordcloud & cluster kaya) ---
    for cluster, templates in CLUSTER_TWEET_TEMPLATES.items():
        for t in templates:
            rows.append({
                "tweet_id": f"T{len(rows)+1:04d}",
                "akun": np.random.choice(CURHATOR_HANDLES),
                "text": t,
                "keyword": np.random.choice(keyword_pool),
                "label_urgensi": np.random.choice(
                    ["Butuh Pertolongan Segera", "Curhat Ringan"], p=[0.4, 0.6]
                ),
                "klaster_penyebab": cluster,
                "likes": np.random.randint(5, 100),
                "replies": np.random.randint(0, 30),
            })

    df = pd.DataFrame(rows)
    df["tanggal"] = _random_dates(len(df))
    df = df.sort_values("tanggal").reset_index(drop=True)
    return df


def generate_support_network(df_tweets, n_edges=260):
    """
    Generate edge list untuk Support System Network (SNA).
    Merepresentasikan interaksi: akun pendukung (supporter) memberi
    balasan/dukungan positif ke akun yang curhat (curhator).

    Return: pandas DataFrame dengan kolom [source, target, weight, reply_text]
    source = akun pendukung, target = akun yang curhat
    """
    edges = []
    curhator_pool = df_tweets["akun"].unique().tolist()

    # Beberapa supporter dibuat jauh lebih aktif (power-law-ish) supaya
    # hasil centrality terlihat jelas & bermakna saat dipresentasikan
    supporter_weights = np.random.dirichlet(
        np.linspace(0.5, 3.0, len(SUPPORTER_HANDLES))[::-1]
    )

    for _ in range(n_edges):
        supporter = np.random.choice(SUPPORTER_HANDLES, p=supporter_weights)
        curhator = np.random.choice(curhator_pool)
        if supporter == curhator:
            continue
        reply = np.random.choice(SUPPORT_REPLY_TEMPLATES)
        edges.append({
            "source": supporter,
            "target": curhator,
            "weight": np.random.randint(1, 4),
            "reply_text": reply,
        })

    # Tambahkan juga sedikit interaksi antar-supporter (saling dukung sesama relawan)
    for _ in range(15):
        a, b = np.random.choice(SUPPORTER_HANDLES, size=2, replace=False)
        edges.append({
            "source": a,
            "target": b,
            "weight": np.random.randint(1, 3),
            "reply_text": "makasih udah bantu jawabin menfess ya kak 🙏",
        })

    return pd.DataFrame(edges)
