# tullkod_model.py
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------
# ENV + TARIC JSON
# ---------------------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TARIC_JSON_PATH = BASE_DIR / "taric_30.json"

with open(TARIC_JSON_PATH, encoding="utf-8") as f:
    taric_data = json.load(f)

# JSON as string passed to the model
taric_text = json.dumps(taric_data, ensure_ascii=False)

# ---------------------------------------------------------------------
# CLASSIFICATION CHAIN (returns a code as text)
# ---------------------------------------------------------------------
system_prompt = """
You are a customs classification assistant.

You are given:
1) The Swedish TARIC/HS classification hierarchy for Chapter 30 (pharmaceutical products) as JSON.
2) A product name.
3) A product description (Swedish).
4) A list of substance names / composition.

Task:
- Identify the most specific correct TARIC code (`code`) from the JSON that matches the product.
- Use ONLY the JSON to decide (do not invent codes).
- Prefer a 10-digit TARIC code when possible.
- If the JSON does not contain a 10-digit match, return the closest most specific code found in the JSON
  (prefer 10 digits; otherwise 8, 6, 4, 2).
- If the product cannot be classified within Chapter 30, return: UNKNOWN.
- Important: Never output descriptions or explanations, just the code (or UNKNOWN).
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "user",
            "JSON:\n{json}\n\n"
            "Produktnamn: {name}\n"
            "Produktbeskrivning: {desc}\n"
            "Produktsammansättning / substanser: {composition}"
        ),
    ]
)

parser = StrOutputParser()

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0,
)

classification_chain = prompt | llm | parser


# ---------------------------------------------------------------------
# NORMALIZATION: enforce digits-only / UNKNOWN
# ---------------------------------------------------------------------
def normalize_code(text: Optional[str]) -> str:
    """
    Normalize the model output into a clean TARIC code or 'UNKNOWN'.

    - Only digits are allowed in the final code.
    - Prefer codes with length 10, 8, 6, 4, 2.
    - If nothing reasonable can be extracted -> 'UNKNOWN'.
    """
    if not text:
        return "UNKNOWN"

    # Keep only digits
    digits = re.sub(r"\D+", "", text)

    if not digits:
        return "UNKNOWN"

    # Accept typical TARIC lengths directly
    for k in (10, 8, 6, 4, 2):
        if len(digits) == k:
            return digits

    # Try to find a 10-digit sequence in the raw text
    m = re.search(r"\b\d{10}\b", text)
    if m:
        return m.group(0)

    # Fallback: if it's all digits and not too long, accept; else UNKNOWN
    return digits if digits.isdigit() and len(digits) <= 12 else "UNKNOWN"


# ---------------------------------------------------------------------
# EXPLANATION CHAIN (short reasoning text in Swedish)
# ---------------------------------------------------------------------
explanation_system_prompt = """
Du är en tullklassificeringsassistent.

Du får:
- TARIC-hierarkin för kapitel 30 som JSON
- Ett produktnamn, en produktbeskrivning, en sammansättning
- Den valda tullkoden (endast siffror eller UNKNOWN)

Uppgift:
- Förklara kortfattat på svenska hur man rimligen kan motivera att just denna kod passar produkten.
- Nämn formen (t.ex. tabletter, lösning, plåster), användning (terapeutisk/profylaktisk/diagnostisk),
  och viktiga kännetecken från beskrivningen som pekar mot kapitel 30 och den aktuella rubriken.
- Svara med 2–4 meningar.
- Om koden är UNKNOWN, förklara kort att produkten inte kan klassificeras i kapitel 30 utifrån uppgifterna.
"""

explanation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", explanation_system_prompt),
        (
            "user",
            "JSON:\n{json}\n\n"
            "Vald tullkod: {code}\n\n"
            "Produktnamn: {name}\n"
            "Produktbeskrivning: {desc}\n"
            "Produktsammansättning / substanser: {composition}"
        ),
    ]
)

explanation_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)

explanation_chain = explanation_prompt | explanation_llm | parser


# ---------------------------------------------------------------------
# PUBLIC API FUNCTIONS
# ---------------------------------------------------------------------
def classify_product(
    product_name: str,
    product_description: str,
    composition: str,
) -> str:
    """
    Call the LLM once for a single product and return a clean TARIC code (or 'UNKNOWN').
    """
    raw = classification_chain.invoke(
        {
            "json": taric_text,
            "name": product_name or "",
            "desc": product_description or "",
            "composition": composition or "",
        }
    )
    return normalize_code(raw)


def add_tullkod_column(
    df: pd.DataFrame,
    name_col: str,
    desc_col: str,
    comp_col: str,
    output_col: str = "Tullkod",
) -> pd.DataFrame:
    """
    Given a dataframe `df` with product information, append a column with TARIC codes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with one row per product.
    name_col : str
        Column name containing product name.
    desc_col : str
        Column name containing product description.
    comp_col : str
        Column name containing composition / substances.
    output_col : str
        Name of the column to create for TARIC codes.

    Returns
    -------
    pd.DataFrame
        Same dataframe with an extra column `output_col`.
    """

    def _classify_row(row: pd.Series) -> str:
        return classify_product(
            str(row.get(name_col, "")),
            str(row.get(desc_col, "")),
            str(row.get(comp_col, "")),
        )

    df[output_col] = df.apply(_classify_row, axis=1)
    return df


def explain_classification(
    product_name: str,
    product_description: str,
    composition: str,
    tullkod: str,
) -> str:
    """
    Returnerar en kort AI-förklaring (2–4 meningar på svenska) till varför
    given tullkod passar produkten (eller varför koden blev UNKNOWN).
    """
    return explanation_chain.invoke(
        {
            "json": taric_text,
            "code": tullkod,
            "name": product_name or "",
            "desc": product_description or "",
            "composition": composition or "",
        }
    )
