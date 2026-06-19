# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import community

# Custom CSS for aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .sidebar .sidebar-content {
        background: #0E1117;
    }
    h1 {
        color: #5DADE2;
    }
    .card {
        border-radius: 15px; 
        background: rgba(255, 255, 255, 0.1); 
        backdrop-filter: blur(10px); 
        padding: 20px; 
        margin: 15px 0;
    }
    .hover-effect:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🧠 Mental Health Gen Z Monitoring Dashboard")

# PANAL 1: Urgency Indicator
st.subheader("Panel 1: Indikator Urgensi Isu")
# Simulate data
tweets_df = pd.DataFrame({
    'tweet': ["I feel burned out from college", 
              "I'm just feeling depressed today", 
              "Need someone to talk to", 
              "Anxiety is overwhelming me", 
              "My mental health is suffering"],
    'urgency': [1, 1, 0, 1, 0]  # 1 for urgent (butuh pertolongan segera), 0 for light (curhat ringan)
})
total_tweets = len(tweets_df)
urgent_tweets = len(tweets_df[tweets_df['urgency'] == 1])
urgent_percentage = (urgent_tweets / total_tweets) * 100

# KPI total tweet
st.metric("Total Tweets", total_tweets)

# Gauge Chart for urgency
st.subheader("Urgency Level Gauge")
gauge_data = {'Urgent': urgent_tweets, 'Light': total_tweets - urgent_tweets}
fig = px.pie(values=gauge_data.values(), names=gauge_data.keys(), title='Klasifikasi Urgensi')
st.plotly_chart(fig)

# Alert status
if urgent_percentage > 40:
    st.warning("🔴 Darurat: Butuh Pertolongan Segera", icon="⚠️")
elif urgent_percentage > 20:
    st.info("🟡 Perlu Perhatian", icon="ℹ️")
else:
    st.success("🟢 Aman", icon="✅")

# PANEL 2: Klasifikasi Tingkat Urgensi
st.subheader("Panel 2: Klasifikasi Tingkat Urgensi")
X_train, X_test, y_train, y_test = train_test_split(tweets_df['tweet'], tweets_df['urgency'], test_size=0.2)
classifier = MultinomialNB()
classifier.fit(X_train.values.reshape(-1, 1), y_train)

y_pred = classifier.predict(X_test.values.reshape(-1, 1))
st.text(classification_report(y_test, y_pred))

# Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix")
st.pyplot(plt)

# Example tweets
st.subheader("Contoh Tweet untuk Setiap Kelas")
example_tweets = {
    1: "I can’t handle the academic pressure anymore.",
    0: "I just want to talk to someone about my day."
}
st.write(example_tweets)

# PANEL 3: Klasterisasi & Trending Topic
st.subheader("Panel 3: Klasterisasi & Trending Topic")
# Simulated data
stress_causes = {"Stress Cause": ["Academic Pressure", "Family Issues", "Financial"], "Count": [50, 30, 20]}
stress_df = pd.DataFrame(stress_causes)

fig = px.pie(stress_df, values='Count', names='Stress Cause', title='Penyebab Stres Terbesar Gen Z')
st.plotly_chart(fig)

wordcloud = WordCloud(width=800, height=400).generate(' '.join(tweets_df['tweet'].tolist()))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# PANAL 4: Support System Network
st.subheader("Panel 4: Support System Network")
# Here we just simulate the data for the network
G = nx.Graph()
G.add_edges_from([("User 1", "Supportive Account A"), ("User 2", "Supportive Account B"),
                  ("User 1", "Supportive Account B"), ("User 2", "Supportive Account C")])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='#5DADE2', edge_color='#8E44AD', node_size=2000)
plt.title("Support Network Graph")
st.pyplot(plt)

# Centrality Analysis
degree_centrality = nx.degree_centrality(G)
most_central = max(degree_centrality, key=degree_centrality.get)
st.write(f"Akun {most_central} merupakan akun dengan interaksi dukungan tertinggi")

# Automatically generated insights
def generate_insights():
    return "Perlu meningkatnya interaksi di platform ini untuk dukungan yang lebih baik."

st.write(generate_insights())

# Run the app with `streamlit run <script_name.py>`
