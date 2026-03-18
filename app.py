import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Retirement Monte Carlo",
    page_icon=""
)

simulator = st.Page("dashboard.py", title="Simulator",  default=True)
docs      = st.Page("pages/docs.py", title="Documentation")

pg = st.navigation(
    {"": [simulator, docs]},
    position="sidebar",
    expanded=True
)

pg.run()