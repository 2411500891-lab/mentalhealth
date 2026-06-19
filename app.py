import streamlit as st

st.set_page_config(
    page_title="Mental Health Dashboard",
    page_icon="🧠",
    layout="wide"
)

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

</style>
""", unsafe_allow_html=True)

st.title("🧠 Mental Health Gen Z Dashboard")

col1,col2,col3=st.columns(3)

with col1:
    st.metric("Total Tweet","250")

with col2:
    st.metric("Darurat","35%")

with col3:
    st.metric("Curhat Ringan","65%")

st.write("Dashboard berhasil tampil 🎉")
