import streamlit as st

st.title("Paper2Code 🚀")

st.write("Upload a research paper PDF to begin (pipeline not implemented yet).")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    st.success("File uploaded successfully! Processing will be added in Week 2+.")
