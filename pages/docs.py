import streamlit as st

with open("README.md") as f:
    st.markdown(f.read())