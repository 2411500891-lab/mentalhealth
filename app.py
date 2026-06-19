import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="Analisis Sentimen Tokopedia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== KAMUS & LEXICON ==================
NORMALIZATION_DICT = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "tdk": "tidak", "engga": "tidak", "nggak": "tidak",
    "brg": "barang", "sdh": "sudah", "dgn": "dengan", "thx": "terima kasih",
    "tks": "terima kasih", "makasih": "terima kasih", "mksh": "terima kasih", "bgt": "banget",
    "kalo": "kalau", "kl": "kalau", "tp": "tapi", "dr": "dari",
    "bs": "bisa", "sy": "saya", "ak": "aku", "aq": "aku",
    "mantul": "mantap", "mantap": "bagus", "jos": "bagus", 
    "ok": "oke", "oke": "bagus", "good": "bagus", "best": "bagus",
    "jelek": "buruk", "parah": "buruk", "ancur": "buruk", "rusak": "buruk",
    "dtg": "datang", "sampe": "sampai", "nyampe": "sampai", "cepet": "cepat",
    "kirim": "pengiriman", "kurir": "pengiriman", "packing": "kemasan",
    "seller": "penjual", "respon": "tanggapan", "bintang": "rating",
    "gan": "juragan", "sis": "kakak", "kak": "kakak", "om": "paman",
    "admin": "penjual", "olshop": "toko online", "bhn": "bahan",
    "adem": "sejuk", "mlar": "melar", "size": "ukuran", "pas": "sesuai",
    "trmksh": "terima kasih", "thankyou": "terima kasih", "recommended": "rekomendasi",
    "recomended": "rekomendasi", "rekomend": "rekomendasi", "mantab": "mantap",
    "bgus": "bagus", "bgs": "bagus", "krn": "karena", "utk": "untuk",
    "buat": "untuk", "krm": "kirim", "pngrmn": "pengiriman", "cpt": "cepat",
    "cpet": "cepat", "lma": "lama", "telat": "terlambat", "mrh": "murah",
    "mhl": "mahal", "ori": "asli", "kw": "palsu", "fake": "palsu",
    "rsk": "rusak", "ssuai": "sesuai", "krg": "kurang", "byk": "banyak",
    "sgt": "sangat", "bener": "benar", "krenn": "keren", "gokil": "keren",
    "bngt": "banget", "bngd": "banget", "puass": "puas", "puasss": "puas",
    "kcwa": "kecewa", "jlek": "jelek", "jelk": "jelek", "bruk": "buruk"
}

POSITIVE_WORDS = {
    "bagus", "baik", "cepat", "rapi", "aman", "sesuai", "mantap", "puas", 
    "oke", "keren", "nyaman", "suka", "awet", "murah", "ramah", "lengkap",
    "lucu", "halus", "lembut", "tebal", "asli", "original", "rekomendasi",
    "top", "memuaskan", "berfungsi", "pas", "cocok", "sejuk",
    "adem", "dingin", "menyerap", "keringat", "elegan", "mewah", "premium",
    "rapih", "kilat", "gesit", "responsif", "sopan", "jujur", "amanah",
    "bonus", "hadiah", "terjangkau", "hemat", "diskon", "promo", "bersih",
    "wangi", "harum", "cantik", "ganteng", "kece", "modis", "trendy",
    "terima kasih", "thanks", "love", "senang", "sukses", "berkualitas",
    "recommended", "sempurna", "istimewa", "terbaik", "favorit", "worth",
    "gercep", "excellent", "seneng", "super", "juara", "aman", "terpercaya"
}

NEGATIVE_WORDS = {
    "jelek", "buruk", "lambat", "lama", "rusak", "cacat", "pecah", "penyok",
    "kecewa", "salah", "beda", "tipis", "kasar", "kotor", "mahal", "bohong",
    "palsu", "kw", "robek", "bolong", "batal", "retur", "komplain", "parah",
    "nyesel", "kurang", "tidak", "panas", "gerah", "gatal", "sempit", 
    "longgar", "kebesaran", "kekecilan", "luntur", "pudar", "kusam",
    "bau", "apek", "lecek", "kusut", "benang", "jahitan", "lepas", "copot",
    "penipuan", "penipu", "lamban", "lelet", "jutek", "galak", "sombong",
    "ribet", "susah", "baret", "gores", "bekas", "mengecewakan",
    "hancur", "ampun", "zonk", "terrible", "awful", "worst", "bad",
    "mengecewakan", "terlambat", "slow", "najis", "sampah", "php",
    "berbeda", "error", "pecah", "sobek", "busuk", "horrible"
}

STOPWORDS = {
    "yang", "di", "dan", "itu", "ini", "dari", "ke", "untuk", "dengan", "nya",
    "saya", "aku", "kami", "kita", "bisa", "ada", "adalah", "juga", "karena",
    "tapi", "namun", "atau", "jadi", "jika", "kalau", "sudah", "lagi", "akan",
    "pada", "masih", "saja", "yg", "dg", "rt", "dgn", "ny", "d", "k",
    "kalo", "biar", "bikin", "bilang", "gak", "ga", "krn", "nih",
    "sih", "si", "tau", "tdk", "tuh", "utk", "ya", "jd", "jgn", "sdh", 
    "aja", "n", "t", "nyg", "hehe", "pen", "u", "nan", "loh", "rt",
    "oleh", "se", "te", "an", "kan", "dia", "mereka", "ia", "telah",
    "sedang", "pernah", "belum", "bukan", "jangan", "bila", "maka",
    "walaupun", "meskipun", "agar", "supaya", "semoga", "dalam", "kepada",
    "terhadap", "antara", "tentang", "hingga", "sambil", "demi", "sebelum",
    "sesudah", "saat", "ketika", "sewaktu", "begitu", "seperti", "bagai",
    "ibarat", "umpama", "laksana", "seolah", "serupa", "macam", "secara",
    "setiap", "seluruh", "semua", "para", "sang", "si", "sri", "banget",
    "sekali", "sangat", "very", "nih", "deh", "dong", "kok", "sih", "lho", "kah", "pun"
}

# ================== FUNGSI PREPROCESSING ==================
def cleaning(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def case_folding(text):
    return text.lower()

def normalization(text):
    words = text.split()
    normalized = [NORMALIZATION_DICT.get(word, word) for word in words]
    return ' '.join(normalized)

def stopword_removal(text):
    words = text.split()
    filtered = [word for word in words if word not in STOPWORDS and len(word) > 2]
    return ' '.join(filtered)

def preprocess_text_step_by_step(text):
    original = str(text)
    step1_cleaning = cleaning(original)
    step2_case_folding = case_folding(step1_cleaning)
    step3_normalization = normalization(step2_case_folding)
    step4_stopword_removal = stopword_removal(step3_normalization)
    
    return {
        'original': original,
        'step1_cleaning': step1_cleaning,
        'step2_case_folding': step2_case_folding,
        'step3_normalization': step3_normalization,
        'step4_stopword_removal': step4_stopword_removal,
        'final': step4_stopword_removal
    }

def preprocess_text(text):
    text = cleaning(text)
    text = case_folding(text)
    text = normalization(text)
    text = stopword_removal(text)
    return text

# ================== PELABELAN HYBRID - ENHANCED (NOVELTY) ==================
def lexicon_sentiment_scoring(text):
    """Lexicon-Based Scoring dengan penghitungan skor sentimen"""
    words = set(text.split())
    pos_count = len(words.intersection(POSITIVE_WORDS))
    neg_count = len(words.intersection(NEGATIVE_WORDS))
    
    total_sentiment_words = pos_count + neg_count
    if total_sentiment_words == 0:
        confidence = 0.0
    else:
        confidence = abs(pos_count - neg_count) / total_sentiment_words
    
    if pos_count > neg_count:
        sentiment = 'Positif'
    elif neg_count > pos_count:
        sentiment = 'Negatif'
    else:
        sentiment = 'Netral'
    
    return sentiment, pos_count, neg_count, confidence

def detect_sarcasm_mismatch(lexicon_label, rating, confidence):
    """Deteksi sarkasme atau kesalahan input rating"""
    try:
        rating = float(rating)
    except:
        return False, "Rating tidak valid"
    
    if lexicon_label == 'Positif' and rating <= 2 and confidence > 0.5:
        return True, "Kemungkinan sarkasme: ulasan positif dengan rating sangat rendah"
    
    if lexicon_label == 'Negatif' and rating >= 4 and confidence > 0.5:
        return True, "Kemungkinan salah input rating: ulasan negatif dengan rating tinggi"
    
    return False, "Konsisten"

def hybrid_labeling_enhanced(text_preprocessed, rating):
    """
    NOVELTY: Sistem Validasi Hybrid yang mengintegrasikan Lexicon-Based Scoring 
    dengan koreksi Rating-Weighted untuk menangani ulasan sarkas atau salah input rating
    """
    lexicon_label, pos_score, neg_score, confidence = lexicon_sentiment_scoring(text_preprocessed)
    
    try:
        rating = float(rating)
    except:
        return {
            'sentiment': lexicon_label,
            'method': 'Lexicon-Only',
            'pos_score': pos_score,
            'neg_score': neg_score,
            'confidence': confidence,
            'is_corrected': False,
            'sarcasm_flag': False,
            'correction_reason': 'Rating tidak valid'
        }
    
    is_suspicious, reason = detect_sarcasm_mismatch(lexicon_label, rating, confidence)
    
    final_sentiment = lexicon_label
    is_corrected = False
    method = 'Lexicon-Only'
    
    if lexicon_label == 'Positif' and rating <= 3:
        if rating <= 2:
            final_sentiment = 'Negatif'
            is_corrected = True
            method = 'Hybrid-Corrected (Strong)'
        else:
            final_sentiment = 'Netral'
            is_corrected = True
            method = 'Hybrid-Corrected (Moderate)'
    
    elif lexicon_label == 'Negatif' and rating >= 4:
        if rating == 5:
            final_sentiment = 'Positif'
            is_corrected = True
            method = 'Hybrid-Corrected (Strong)'
        else:
            final_sentiment = 'Netral'
            is_corrected = True
            method = 'Hybrid-Corrected (Moderate)'
    
    elif lexicon_label == 'Netral':
        if rating >= 4:
            final_sentiment = 'Positif'
            method = 'Rating-Based'
        elif rating <= 2:
            final_sentiment = 'Negatif'
            method = 'Rating-Based'
        else:
            method = 'Lexicon-Rating-Agree'
    else:
        method = 'Lexicon-Rating-Agree'
    
    return {
        'sentiment': final_sentiment,
        'method': method,
        'pos_score': pos_score,
        'neg_score': neg_score,
        'confidence': confidence,
        'is_corrected': is_corrected,
        'sarcasm_flag': is_suspicious,
        'correction_reason': reason,
        'original_lexicon': lexicon_label,
        'rating': rating
    }

# ================== FUNGSI DETEKSI KOLOM ==================
def detect_rating_column(df):
    rating_candidates = ['rating', 'bintang', 'star', 'rate', 'nilai', 'score']
    
    for col in df.columns:
        if col.lower() in rating_candidates:
            return col
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].min() >= 1 and df[col].max() <= 5:
                return col
    return None

def detect_product_column(df):
    product_candidates = ['nama_produk', 'product_name', 'produk', 'product', 'nama', 'name', 'item']
    
    for col in df.columns:
        if col.lower() in product_candidates:
            return col
    return None

def detect_sales_column(df):
    sales_candidates = ['jumlah_terjual', 'terjual', 'sold', 'sales', 'qty_sold', 'quantity_sold']
    
    for col in df.columns:
        if col.lower() in sales_candidates:
            return col
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].min() >= 0 and col.lower() not in ['rating', 'bintang', 'star']:
                return col
    return None

# ================== ANALISIS TOP PRODUCTS (NOVELTY) ==================
def analyze_top_products(df, product_col, rating_col, sales_col=None, top_n=10):
    """
    NOVELTY: Analisis Top Products untuk insight bisnis
    """
    results = {}
    
    # 1. Produk Terbanyak Diulas
    product_review_counts = df[product_col].value_counts().head(top_n)
    results['most_reviewed'] = pd.DataFrame({
        'Produk': product_review_counts.index,
        'Jumlah Ulasan': product_review_counts.values
    })
    
    # 2. Produk dengan Rating Tertinggi (minimal 5 ulasan)
    product_ratings = df.groupby(product_col).agg({
        rating_col: ['mean', 'count']
    }).reset_index()
    product_ratings.columns = ['Produk', 'Rata-rata Rating', 'Jumlah Ulasan']
    product_ratings = product_ratings[product_ratings['Jumlah Ulasan'] >= 5]
    product_ratings = product_ratings.sort_values('Rata-rata Rating', ascending=False).head(top_n)
    results['highest_rated'] = product_ratings
    
    # 3. Produk dengan Sentimen Positif Terbanyak
    positive_df = df[df['sentiment'] == 'Positif']
    positive_counts = positive_df[product_col].value_counts().head(top_n)
    results['most_positive'] = pd.DataFrame({
        'Produk': positive_counts.index,
        'Jumlah Ulasan Positif': positive_counts.values
    })
    
    # 4. Produk Terlaris
    if sales_col:
        try:
            product_sales = df.groupby(product_col)[sales_col].first().sort_values(ascending=False).head(top_n)
            results['best_selling'] = pd.DataFrame({
                'Produk': product_sales.index,
                'Jumlah Terjual': product_sales.values
            })
        except:
            results['best_selling'] = None
    else:
        results['best_selling'] = None
    
    # 5. Analisis Sentimen per Produk
    top_products = product_review_counts.head(10).index
    sentiment_by_product = df[df[product_col].isin(top_products)].groupby([product_col, 'sentiment']).size().unstack(fill_value=0)
    results['sentiment_distribution'] = sentiment_by_product
    
    return results

# ================== UI UTAMA ==================
st.title("📊 Analisis Sentimen Ulasan Tokopedia")
st.markdown("""
**Metode Penelitian:** Pendekatan Kuantitatif dengan Waterfall yang Dimodifikasi untuk Data Science

**🆕 Kebaruan (Novelty):**
1. **Sistem Validasi Hybrid**: Integrasi Lexicon-Based Scoring dengan koreksi Rating-Weighted untuk menangani sarkasme dan salah input rating
2. **Analisis Top Products**: Identifikasi produk terlaris dan terbanyak diulas sebagai insight bisnis strategis
""")

with st.sidebar:
    st.header("📂 1. Upload Data")
    uploaded_file = st.file_uploader("Upload file CSV ulasan Tokopedia", type=["csv"])
    
    st.divider()
    st.header("⚙️ 2. Pengaturan Model")
    st.info("**Split Data:** 80% Training, 20% Testing")
    st.caption("Model: SVM dengan kernel Linear")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset berhasil dimuat: **{len(df)}** baris data")
    
    rating_col = detect_rating_column(df)
    product_col = detect_product_column(df)
    sales_col = detect_sales_column(df)
    
    if rating_col is None:
        st.error("⚠️ Kolom 'rating' tidak ditemukan!")
        st.stop()
    else:
        st.info(f"✅ Kolom rating: **{rating_col}**")
    
    if product_col:
        st.info(f"✅ Kolom produk: **{product_col}**")
    if sales_col:
        st.info(f"✅ Kolom penjualan: **{sales_col}**")
    
    st.subheader("Konfigurasi Kolom Dataset")
    text_col = st.selectbox("Pilih Kolom Ulasan (Teks):", df.columns, index=0)
    
    st.subheader("📝 Tahap Preprocessing Data")
    
    with st.expander("🔍 Lihat Detail Proses Preprocessing"):
        st.markdown("""
        **Tahapan Preprocessing:**
        1. **Cleaning** - Menghapus tanda baca, angka, karakter non-alfanumerik
        2. **Case Folding** - Mengubah teks menjadi huruf kecil
        3. **Normalization** - Mengubah kata gaul menjadi kata baku
        4. **Stopword Removal** - Menghapus kata umum
        """)
        
        sample_idx = st.number_input("Pilih indeks baris:", min_value=0, max_value=len(df)-1, value=0)
        sample_text = df[text_col].iloc[sample_idx]
        
        st.write("**Teks Original:**")
        st.code(sample_text)
        
        steps = preprocess_text_step_by_step(sample_text)
        
        st.write("**Tahap 1 - Cleaning:**")
        st.code(steps['step1_cleaning'])
        
        st.write("**Tahap 2 - Case Folding:**")
        st.code(steps['step2_case_folding'])
        
        st.write("**Tahap 3 - Normalization:**")
        st.code(steps['step3_normalization'])
        
        st.write("**Tahap 4 - Stopword Removal:**")
        st.code(steps['step4_stopword_removal'])
    
    with st.spinner("⏳ Melakukan preprocessing dan pelabelan hybrid..."):
        df['step1_cleaning'] = df[text_col].astype(str).apply(cleaning)
        df['step2_case_folding'] = df['step1_cleaning'].apply(case_folding)
        df['step3_normalization'] = df['step2_case_folding'].apply(normalization)
        df['step4_stopword_removal'] = df['step3_normalization'].apply(stopword_removal)
        df['text_preprocessed'] = df['step4_stopword_removal']
        
        hybrid_results = df.apply(lambda row: hybrid_labeling_enhanced(row['text_preprocessed'], row[rating_col]), axis=1)
        
        df['sentiment'] = hybrid_results.apply(lambda x: x['sentiment'])
        df['labeling_method'] = hybrid_results.apply(lambda x: x['method'])
        df['pos_score'] = hybrid_results.apply(lambda x: x['pos_score'])
        df['neg_score'] = hybrid_results.apply(lambda x: x['neg_score'])
        df['confidence'] = hybrid_results.apply(lambda x: x['confidence'])
        df['is_corrected'] = hybrid_results.apply(lambda x: x['is_corrected'])
        df['sarcasm_flag'] = hybrid_results.apply(lambda x: x['sarcasm_flag'])
    
    st.success("✅ Preprocessing & Pelabelan Hybrid Enhanced selesai!")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Total Ulasan", len(df))
    col_stat2.metric("Label Dikoreksi", df['is_corrected'].sum())
    col_stat3.metric("Deteksi Sarkasme", df['sarcasm_flag'].sum())
    col_stat4.metric("Confidence Rata-rata", f"{df['confidence'].mean():.2f}")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Distribusi Sentimen", 
        "🔬 Validasi Hybrid",
        "🏆 Top Products",
        "☁️ WordCloud", 
        "📋 Tabel Interaktif", 
        "🤖 Model SVM",
        "🔮 Prediksi Baru"
    ])
    
    with tab1:
        st.subheader("📊 Grafik Distribusi Sentimen")
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        col_metric1.metric("Total Ulasan", len(df))
        col_metric2.metric("Rata-rata Rating", f"{df[rating_col].mean():.2f} ⭐")
        sentiment_dist = df['sentiment'].value_counts()
        col_metric3.metric("Sentimen Dominan", sentiment_dist.index[0])
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            sentiment_counts = df['sentiment'].value_counts()
            ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                      startangle=90, colors=colors)
            ax_pie.set_title('Proporsi Sentimen', fontsize=14, fontweight='bold')
            st.pyplot(fig_pie)
        
        with col_viz2:
            fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='sentiment', order=['Positif', 'Negatif', 'Netral'], 
                         palette=colors, ax=ax_bar)
            ax_bar.set_title('Distribusi Sentimen', fontsize=14, fontweight='bold')
            ax_bar.set_xlabel('Sentimen')
            ax_bar.set_ylabel('Jumlah')
            for container in ax_bar.containers:
                ax_bar.bar_label(container)
            st.pyplot(fig_bar)
        
        st.subheader("Sentimen Berdasarkan Rating")
        fig_cross, ax_cross = plt.subplots(figsize=(10, 5))
        ct = pd.crosstab(df[rating_col], df['sentiment'])
        ct.plot(kind='bar', ax=ax_cross, color=colors)
        ax_cross.set_title('Distribusi Sentimen per Rating', fontsize=14, fontweight='bold')
        ax_cross.set_xlabel('Rating')
        ax_cross.set_ylabel('Jumlah')
        ax_cross.legend(title='Sentimen')
        plt.xticks(rotation=0)
        st.pyplot(fig_cross)
    
    with tab2:
        st.subheader("🔬 Analisis Sistem Validasi Hybrid")
        
        st.markdown("""
        **Sistem Validasi Hybrid** mengintegrasikan:
        - **Lexicon-Based Scoring**: Menghitung skor kata positif dan negatif
        - **Rating-Weighted Correction**: Koreksi label berdasarkan inkonsistensi rating
        - **Sarcasm Detection**: Deteksi ulasan sarkastik atau salah input rating
        """)
        
        col_h1, col_h2, col_h3 = st.columns(3)
        
        method_counts = df['labeling_method'].value_counts()
        col_h1.metric("Lexicon-Only", method_counts.get('Lexicon-Only', 0) + method_counts.get('Lexicon-Rating-Agree', 0))
        col_h2.metric("Hybrid-Corrected", df['is_corrected'].sum())
        col_h3.metric("Rating-Based", method_counts.get('Rating-Based', 0))
        
        st.subheader("Distribusi Metode Pelabelan")
        fig_method, ax_method = plt.subplots(figsize=(10, 6))
        method_counts.plot(kind='bar', ax=ax_method, color='steelblue')
        ax_method.set_title('Metode Pelabelan yang Digunakan', fontsize=14, fontweight='bold')
        ax_method.set_xlabel('Metode')
        ax_method.set_ylabel('Jumlah')
        ax_method.set_xticklabels(ax_method.get_xticklabels(), rotation=45, ha='right')
        for container in ax_method.containers:
            ax_method.bar_label(container)
        plt.tight_layout()
        st.pyplot(fig_method)
        
        st.subheader("Contoh Kasus Koreksi Hybrid")
        corrected_df = df[df['is_corrected'] == True].head(10)
        if len(corrected_df) > 0:
            display_corrected = corrected_df[[text_col, rating_col, 'sentiment', 'labeling_method', 'confidence']].copy()
            display_corrected.columns = ['Ulasan', 'Rating', 'Sentimen Akhir', 'Metode', 'Confidence']
            st.dataframe(display_corrected, use_container_width=True)
        else:
            st.info("Tidak ada kasus koreksi pada dataset ini")
        
        st.subheader("Deteksi Sarkasme/Mismatch")
        sarcasm_df = df[df['sarcasm_flag'] == True].head(10)
        if len(sarcasm_df) > 0:
            display_sarcasm = sarcasm_df[[text_col, rating_col, 'sentiment', 'pos_score', 'neg_score']].copy()
            display_sarcasm.columns = ['Ulasan', 'Rating', 'Sentimen', 'Skor Positif', 'Skor Negatif']
            st.dataframe(display_sarcasm, use_container_width=True)
        else:
            st.info("Tidak terdeteksi kasus sarkasme/mismatch pada dataset ini")
    
    with tab3:
        st.subheader("🏆 Analisis Top Products (Novelty)")
        
        if product_col:
            top_n = st.slider("Jumlah Top Products:", 5, 20, 10)
            
            with st.spinner("⏳ Menganalisis top products..."):
                results = analyze_top_products(df, product_col, rating_col, sales_col, top_n)
            
            tab_p1, tab_p2, tab_p3, tab_p4, tab_p5 = st.tabs([
                "📊 Terbanyak Diulas",
                "⭐ Rating Tertinggi", 
                "Sentimen Positif",
                "🛒 Terlaris",
                "📈 Distribusi Sentimen"
            ])
            
            with tab_p1:
                st.subheader(f"Top {top_n} Produk Terbanyak Diulas")
                st.dataframe(results['most_reviewed'], use_container_width=True, hide_index=True)
                
                fig_rev, ax_rev = plt.subplots(figsize=(10, 6))
                ax_rev.barh(results['most_reviewed']['Produk'], results['most_reviewed']['Jumlah Ulasan'], color='#3498db')
                ax_rev.set_xlabel('Jumlah Ulasan')
                ax_rev.set_title(f'Top {top_n} Produk Terbanyak Diulas', fontweight='bold')
                ax_rev.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig_rev)
            
            with tab_p2:
                st.subheader(f"Top {top_n} Produk Rating Tertinggi (Min. 5 ulasan)")
                st.dataframe(results['highest_rated'], use_container_width=True, hide_index=True)
                
                fig_rat, ax_rat = plt.subplots(figsize=(10, 6))
                ax_rat.barh(results['highest_rated']['Produk'], results['highest_rated']['Rata-rata Rating'], color='#f39c12')
                ax_rat.set_xlabel('Rata-rata Rating')
                ax_rat.set_xlim(0, 5)
                ax_rat.set_title(f'Top {top_n} Produk dengan Rating Tertinggi', fontweight='bold')
                ax_rat.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig_rat)
            
            with tab_p3:
                st.subheader(f"Top {top_n} Produk Sentimen Positif Terbanyak")
                st.dataframe(results['most_positive'], use_container_width=True, hide_index=True)
                
                fig_pos, ax_pos = plt.subplots(figsize=(10, 6))
                ax_pos.barh(results['most_positive']['Produk'], results['most_positive']['Jumlah Ulasan Positif'], color='#2ecc71')
                ax_pos.set_xlabel('Jumlah Ulasan Positif')
                ax_pos.set_title(f'Top {top_n} Produk dengan Sentimen Positif Terbanyak', fontweight='bold')
                ax_pos.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig_pos)
            
            with tab_p4:
                if results['best_selling'] is not None:
                    st.subheader(f"Top {top_n} Produk Terlaris")
                    st.dataframe(results['best_selling'], use_container_width=True, hide_index=True)
                    
                    fig_sell, ax_sell = plt.subplots(figsize=(10, 6))
                    ax_sell.barh(results['best_selling']['Produk'], results['best_selling']['Jumlah Terjual'], color='#e74c3c')
                    ax_sell.set_xlabel('Jumlah Terjual')
                    ax_sell.set_title(f'Top {top_n} Produk Terlaris', fontweight='bold')
                    ax_sell.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig_sell)
                else:
                    st.warning("⚠️ Data penjualan tidak tersedia. Pastikan dataset memiliki kolom jumlah terjual.")
            
            with tab_p5:
                st.subheader("Distribusi Sentimen per Produk Top 10")
                st.dataframe(results['sentiment_distribution'], use_container_width=True)
                
                fig_dist, ax_dist = plt.subplots(figsize=(12, 6))
                results['sentiment_distribution'].plot(kind='bar', stacked=True, ax=ax_dist, 
                                                       color=['#2ecc71', '#e74c3c', '#95a5a6'])
                ax_dist.set_title('Distribusi Sentimen Top 10 Produk', fontsize=14, fontweight='bold')
                ax_dist.set_xlabel('Produk')
                ax_dist.set_ylabel('Jumlah Ulasan')
                ax_dist.legend(title='Sentimen')
                ax_dist.set_xticklabels(ax_dist.get_xticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_dist)
        else:
            st.warning("⚠️ Kolom produk tidak terdeteksi. Analisis Top Products memerlukan kolom nama produk.")
    
    with tab4:
        st.subheader("☁️ WordCloud per Sentimen")
        
        sentiment_filter = st.selectbox("Pilih Sentimen:", ['Positif', 'Negatif', 'Netral'])
        
        filtered_text = ' '.join(df[df['sentiment'] == sentiment_filter]['text_preprocessed'])
        
        if filtered_text.strip():
            colormap = 'RdYlGn' if sentiment_filter == 'Positif' else 'Reds' if sentiment_filter == 'Negatif' else 'Greys'
            wc = WordCloud(width=1200, height=500, background_color='white',
                          colormap=colormap, max_words=100).generate(filtered_text)
            
            fig_wc, ax_wc = plt.subplots(figsize=(14, 7))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            ax_wc.set_title(f'WordCloud - Sentimen {sentiment_filter}', fontsize=16, fontweight='bold')
            st.pyplot(fig_wc)
        else:
            st.warning(f"⚠️ Tidak ada data untuk sentimen {sentiment_filter}")
    
    with tab5:
        st.subheader("📋 Tabel Interaktif Hasil Analisis")
        
        view_option = st.radio("Pilih Tampilan:", 
                               ["Hasil Akhir + Hybrid Info", "Tahapan Preprocessing"],
                               horizontal=True)
        
        sentiment_filter_table = st.multiselect(
            "Filter Sentimen:", 
            df['sentiment'].unique(), 
            default=list(df['sentiment'].unique())
        )
        
        display_df = df[df['sentiment'].isin(sentiment_filter_table)]
        
        if view_option == "Hasil Akhir + Hybrid Info":
            display_cols = [text_col, 'text_preprocessed', rating_col, 'sentiment', 
                           'labeling_method', 'confidence', 'is_corrected', 'sarcasm_flag']
            if product_col:
                display_cols.insert(0, product_col)
        else:
            display_cols = [text_col, 'step1_cleaning', 'step2_case_folding', 
                           'step3_normalization', 'step4_stopword_removal', 
                           rating_col, 'sentiment']
        
        st.dataframe(display_df[display_cols], use_container_width=True, height=400)
        
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Analisis", csv, "hasil_analisis_sentimen.csv", "text/csv")
    
    with tab6:
        st.subheader("🤖 Pemodelan dengan Support Vector Machine (SVM)")
        st.markdown("""
        **Ekstraksi Fitur:** TF-IDF (Term Frequency–Inverse Document Frequency)  
        **Algoritma:** Support Vector Machine dengan kernel Linear  
        **Split Data:** 80:20 (Training:Testing)
        """)
        
        if st.button("🚀 Latih Model SVM", type="primary"):
            with st.spinner("⏳ Melatih model SVM..."):
                vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
                X = vectorizer.fit_transform(df['text_preprocessed'])
                y = df['sentiment']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                svm_model = SVC(kernel='linear', random_state=42)
                svm_model.fit(X_train, y_train)
                
                y_pred = svm_model.predict(X_test)
                
                st.session_state['svm_model'] = svm_model
                st.session_state['vectorizer'] = vectorizer
                
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred, labels=['Positif', 'Negatif', 'Netral'])
                cr = classification_report(y_test, y_pred, output_dict=True)
                
                st.success(f"✅ Model berhasil dilatih dengan **Accuracy: {accuracy:.4f}**")
                
                col_eval1, col_eval2 = st.columns(2)
                
                with col_eval1:
                    st.subheader("Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Positif', 'Negatif', 'Netral'],
                               yticklabels=['Positif', 'Negatif', 'Netral'], ax=ax_cm)
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('Actual')
                    ax_cm.set_title('Confusion Matrix', fontweight='bold')
                    st.pyplot(fig_cm)
                
                with col_eval2:
                    st.subheader("Classification Report")
                    report_df = pd.DataFrame(cr).transpose()
                    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                
                st.subheader("Performa Model per Sentimen")
                metrics_data = []
                for label in ['Positif', 'Negatif', 'Netral']:
                    if label in cr:
                        metrics_data.append({
                            'Sentimen': label,
                            'Precision': f"{cr[label]['precision']:.4f}",
                            'Recall': f"{cr[label]['recall']:.4f}",
                            'F1-Score': f"{cr[label]['f1-score']:.4f}",
                            'Support': int(cr[label]['support'])
                        })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with tab7:
        st.subheader("🔮 Prediksi Sentimen Ulasan Baru")
        
        if 'svm_model' not in st.session_state:
            st.warning("⚠️ Model belum dilatih! Silakan latih model di tab **Model SVM** terlebih dahulu.")
        else:
            st.info("💡 Masukkan teks ulasan baru untuk memprediksi sentimennya")
            
            new_review = st.text_area("Masukkan ulasan:", 
                                      placeholder="Contoh: Barang bagus banget, cepat sampai!",
                                      height=100)
            
            predict_button = st.button("🚀 Prediksi Sentimen", type="primary")
            
            if predict_button and new_review:
                with st.spinner("⏳ Memproses ulasan..."):
                    preprocessed = preprocess_text(new_review)
                    
                    vectorizer = st.session_state['vectorizer']
                    new_review_tfidf = vectorizer.transform([preprocessed])
                    
                    svm_model = st.session_state['svm_model']
                    prediction = svm_model.predict(new_review_tfidf)[0]
                    
                    st.divider()
                    st.subheader("Hasil Prediksi")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.write("**Teks Original:**")
                        st.info(new_review)
                        
                        st.write("**Teks Setelah Preprocessing:**")
                        st.code(preprocessed)
                    
                    with col_res2:
                        if prediction == 'Positif':
                            st.success(f"### Sentimen: {prediction}")
                        elif prediction == 'Negatif':
                            st.error(f"### Sentimen: {prediction}")
                        else:
                            st.warning(f"### Sentimen: {prediction}")
                        
                        words = set(preprocessed.split())
                        pos_words = words.intersection(POSITIVE_WORDS)
                        neg_words = words.intersection(NEGATIVE_WORDS)
                        
                        st.write("**Kata Kunci Terdeteksi:**")
                        if pos_words:
                            st.write(f"✅ Positif: {', '.join(list(pos_words)[:5])}")
                        if neg_words:
                            st.write(f"❌ Negatif: {', '.join(list(neg_words)[:5])}")
            
            elif predict_button and not new_review:
                st.error("⚠️ Silakan masukkan teks ulasan terlebih dahulu!")

else:
    st.info("👈 upload file CSV dataset ulasan Tokopedia di sidebar untuk memulai analisis")
    
    st.subheader("👥 Anggota Kelompok")
    st.markdown("""
    - **Shelly Ananda** (2411500891)
    - **M. Richo Irsyad F.** (2411500958)
    - **Nazif Hamza Effendy** (2411501527)
     """)
  
    
    

