# MFMA2_app.py

import streamlit as st
#from app.main import run_app

import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

st.set_page_config(page_title="MFMA Dataset Wizard", layout="wide")

img_base64 = image_to_base64("CBG_sicpa_logo.png")

st.markdown(
    f"""
    <div style="text-align: center; margin-top: 20px; margin-bottom: 10px;">
        <img src="data:image/png;base64,{img_base64}" width="250" style="opacity: 0.95;"/>
    </div>
    <div style="text-align: center; font-size: 2.2rem; font-weight: 600; margin-bottom: 5px;">
        MFMA Dataset Wizard
    </div>
    <div style="text-align: center; font-size: 1rem; color: #666;">
        Multi-Feature Membership Analysis for regression models<br>
        <span style="font-size: 0.9rem;">@ Tsachi Cahana, 19.06.2025</span>
    </div>
    <hr style="margin-top: 20px; margin-bottom: 10px;">
    """,
    unsafe_allow_html=True
)

import importlib

main_module = importlib.import_module("dataset_app.main")
main_module.run_app()
