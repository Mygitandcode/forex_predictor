import streamlit as st
st.title("Test")
st.write("Streamlit works!")
import requests
try:
    r = requests.get("http://localhost:8000/health", timeout=5)
    st.success(f"API OK: {r.json()}")
except Exception as e:
    st.error(f"API FAIL: {e}")
