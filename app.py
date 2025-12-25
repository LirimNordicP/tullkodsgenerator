# app.py
import io
import os

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from tullkod_model import (
    classify_product,
    add_tullkod_column,
    explain_classification,
)

# --------------------------------------------------------------------
# BASIC SETUP
# --------------------------------------------------------------------
load_dotenv()
APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")

st.set_page_config(
    page_title="Swedish customs-code generator",
    page_icon="💊",
    layout="wide",
)

# --------------------------------------------------------------------
# GLOBAL STYLING (LIGHT BABY-BLUE THEME)
# --------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --primary-color: #2563eb;
        --primary-soft: rgba(37, 99, 235, 0.10);
        --baby-blue: #e9f4ff;
        --baby-blue-border: #c5ddff;
        --baby-blue-border-strong: #9cc6ff;
        --card-bg: #ffffff;
        --bg-gradient-start: #eaf4ff;
        --bg-gradient-end: #f9faff;
        --shadow-soft: 0 18px 40px rgba(15, 23, 42, 0.08);
        --text-strong: #0f172a;
        --text-muted: #64748b;
    }

    /* PAGE BACKGROUND */
    .stApp {
        background: linear-gradient(
            145deg,
            var(--bg-gradient-start),
            var(--bg-gradient-end)
        ) !important;
    }

    /* TYPOGRAPHY */
    h1, h2, h3, h4 {
        color: var(--text-strong) !important;
        font-weight: 700 !important;
    }

    p, span, label, .stMarkdown {
        color: var(--text-muted) !important;
    }

    /* CARDS */
    .tull-card {
        background: var(--card-bg) !important;
        padding: 1.6rem 1.9rem;
        border-radius: 18px;
        box-shadow: var(--shadow-soft);
        border: 1px solid rgba(148, 163, 184, 0.23);
    }

    .tull-card-soft {
        background: rgba(255, 255, 255, 0.85) !important;
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.20);
    }

    /* BUTTONS */
    .stButton button {
        border-radius: 999px !important;
        padding: 0.55rem 1.6rem !important;
        font-weight: 600 !important;
        border: none !important;
        background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.38) !important;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 36px rgba(37, 99, 235, 0.48) !important;
    }

    /* Make sure button text is white */
    .stButton button,
    .stButton button * {
        color: #ffffff !important;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 999px !important;
        padding: 0.3rem 1.1rem !important;
        border: 1px solid rgba(148, 163, 184, 0.35);
        color: var(--text-muted);
    }

    .stTabs [aria-selected="true"] {
        background: var(--baby-blue) !important;
        border: 1px solid var(--baby-blue-border-strong) !important;
        color: #1d4ed8 !important;
        font-weight: 600 !important;
    }

    /* INPUTS — BABY BLUE THEME */

    /* reset outer wrappers */
    .stTextInput > div,
    .stTextArea > div,
    .stSelectbox > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* input shell */
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stSelectbox > div > div {
        background-color: var(--baby-blue) !important;
        border-radius: 14px !important;
        border: 1px solid var(--baby-blue-border) !important;
        box-shadow: none !important;
        color: var(--text-strong) !important;
    }

    /* actual inputs */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background-color: transparent !important;
        border: none !important;
        color: var(--text-strong) !important;
    }

    /* focus */
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus,
    .stSelectbox > div > div:focus-within {
        outline: none !important;
        border: 1.6px solid var(--baby-blue-border-strong) !important;
        background-color: rgba(173, 216, 255, 0.45) !important;
        box-shadow: 0 0 0 1px rgba(156, 198, 255, 0.55) !important;
    }

    /* FILE UPLOADER — baby blue, no dark bar */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: var(--baby-blue) !important;
        border-radius: 14px !important;
        border: 1px dashed var(--baby-blue-border-strong) !important;
        box-shadow: none !important;
        padding: 0.75rem !important;
    }

    /* Make everything inside the dropzone transparent so no dark strip remains */
    div[data-testid="stFileUploaderDropzone"] * {
        background-color: transparent !important;
        box-shadow: none !important;
    }

    /* Optional: lighten the 'Browse files' button border/text if needed */
    div[data-testid="stFileUploaderDropzone"] button {
        border-radius: 999px !important;
        border: 1px solid var(--baby-blue-border-strong) !important;
        color: var(--text-strong) !important;
    }


    /* HIDE STREAMLIT DEFAULT ELEMENTS */
    #MainMenu, footer {
        visibility: hidden;
    }

    /* === HARD RESET FOR FILE UPLOADER === */
    div[data-testid="stFileUploader"] {
        background: transparent !important;
        box-shadow: none !important;
    }

    div[data-testid="stFileUploader"] * {
        background: transparent !important;
        box-shadow: none !important;
    }

    /* FILE UPLOADER — baby-blue with visible outline */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: var(--baby-blue) !important;
        border-radius: 14px !important;
        border: 2px dashed #7aa7e9 !important;
        box-shadow: 0 0 0 1px rgba(122, 167, 233, 0.35) !important;
        padding: 0.9rem !important;
        color: var(--text-muted) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

# --------------------------------------------------------------------
# LOGIN
# --------------------------------------------------------------------
def login():
    """Simple username/password login using .env credentials."""
    st.markdown("<br><br>", unsafe_allow_html=True)
    cols = st.columns([1, 2, 1])

    with cols[1]:
        st.markdown(
            """
            <div class="tull-card">
                <h2 style="margin-bottom: 0.4rem;">🔐 Customs-code generator</h2>
                <p style="margin-top: 0; font-size: 0.95rem;">
                    Log in to use the tool for classification of pharmaceutical products.
                </p>
            """,
            unsafe_allow_html=True,
        )

        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        login_button = st.button("Log in")

        if login_button:
            if user == APP_USERNAME and pwd == APP_PASSWORD:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Wrong username.")

        st.markdown("</div>", unsafe_allow_html=True)


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# --------------------------------------------------------------------
# MAIN APP CONTENT
# --------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="tull-card">
        <h1 style="margin-bottom: 0.3rem;">💊 Customs code generator</h1>
        <p style="margin-top: 0; font-size: 0.97rem;">
            Provide product information manually or upload an excel-file in order to generate customs codes.
            (farmaceutiska produkter Kapitel 30).
        </p>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_bulk = st.tabs(["Single product", "Excel-upload"])

# --------------------------------------------------------------------
# TAB 1: Single product classification
# --------------------------------------------------------------------
with tab_single:
    st.markdown("<br>", unsafe_allow_html=True)
    #st.markdown('<div class="tull-card-soft">', unsafe_allow_html=True)
    st.subheader("Single product")

    col1, col2 = st.columns(2)

    with col1:
        product_name = st.text_input("Product name", value="")
        product_description = st.text_area(
            "Product description",
            height=150,
            value="",
        )

    with col2:
        composition = st.text_area(
            "Product composition / substance name",
            height=150,
            help="T.ex. verksamma substanser, hjälpämnen, styrkor osv.",
        )

    if st.button("Generate code", type="primary"):
        if not (product_name or product_description or composition):
            st.warning("Fill in at least one field before generating.")
        else:
            with st.spinner("Classifying product..."):
                tullkod = classify_product(
                    product_name=product_name,
                    product_description=product_description,
                    composition=composition,
                )

            st.success(f"Customs code: **{tullkod}**")

            # AI explanation
            with st.spinner("Explaining the classification..."):
                explanation = explain_classification(
                    product_name=product_name,
                    product_description=product_description,
                    composition=composition,
                    tullkod=tullkod,
                )

            if tullkod != "UNKNOWN":
                st.markdown(f"**AI-explanation:** {explanation}")
            else:
                st.markdown(
                    "**AI-explanation:** Produkten kunde inte säkert klassificeras inom "
                    "kapitel 30 utifrån den information som angavs."
                )
            st.markdown(
                "[🔗 Öppna Tulltaxan för att kontrollera koden]"
                "(https://tulltaxan.tullverket.se/ite-tariff-public/#/taric/nomenclature/sbn?sd=2025-10-22&d=I&l=sv&ql=sv)"
            )


    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# TAB 2: Bulk classification via Excel
# --------------------------------------------------------------------
with tab_bulk:
    st.markdown("<br>", unsafe_allow_html=True)
    #st.markdown('<div class="tull-card-soft">', unsafe_allow_html=True)
    st.subheader("Excel-upload")

    st.markdown(
        """
        Upload an Excel file with one row per product.
        It is good (but not required) if you have separate columns for, for example:

        Product name (e.g., `Product name`)
        Product description (e.g., `Product description`)
        Product composition / substance name (e.g., `Product composition`)

        In the next step, you will select which columns to use for the name, description, and composition yourself.
        """,
        unsafe_allow_html=False,
    )

    uploaded_file = st.file_uploader(
        "Upload an Excel-file (.xlsx) with product information",
        type=["xlsx"],
    )

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        st.markdown("#### Data preview")
        st.dataframe(df.head())

        if df.empty:
            st.warning("The uploaded file seems to be empty.")
        else:
            st.markdown("#### Select columns for classification.")

            name_col = st.selectbox(
                "Column for product name",
                options=df.columns,
                index=0,
            )

            desc_default = 1 if len(df.columns) > 1 else 0
            comp_default = 2 if len(df.columns) > 2 else desc_default

            desc_col = st.selectbox(
                "Column for product description",
                options=df.columns,
                index=desc_default,
            )

            comp_col = st.selectbox(
                "Column for product composition / substances",
                options=df.columns,
                index=comp_default,
            )

            if st.button("Generate customs codes for all records", type="primary"):
                with st.spinner("Classifying all products..."):
                    result_df = add_tullkod_column(
                        df.copy(),
                        name_col=name_col,
                        desc_col=desc_col,
                        comp_col=comp_col,
                        output_col="Customs code",
                    )

                st.success("Classification is completed!")
                st.markdown("#### Preview with customs code")
                st.dataframe(result_df.head())

                buffer = io.BytesIO()
                result_df.to_excel(buffer, index=False)
                buffer.seek(0)

                st.download_button(
                    label="Download the results (Excel)",
                    data=buffer,
                    file_name="tullkoder_resultat.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )

                st.markdown(
                    "[🔗 Öppna Tulltaxan för att kontrollera koderna]"
                    "(https://tulltaxan.tullverket.se/ite-tariff-public/#/taric/nomenclature/sbn?sd=2025-10-22&d=I&l=sv&ql=sv)"
                )
    else:
        st.info("Upload an excel file to proceed.")

    st.markdown("</div>", unsafe_allow_html=True)
