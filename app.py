import sys
import re
from pathlib import Path
from urllib.parse import urlparse
import base64

import pandas as pd
import streamlit as st
import torch

# Allow app.py to import files from src/
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
ASSETS_DIR = ROOT_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "VeriqURL_logo.png"
sys.path.append(str(SRC_DIR))

#=========================
# For logo use
def image_to_base64(image_path: Path) -> str:
    """
    Convert local image to base64 so it can be embedded inside custom HTML.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

from predict_compare import (
    load_rf_model,
    load_char_vocab,
    load_tcn_model,
    predict_rf,
    predict_tcn,
    get_final_decision,
)

from explanation_rules import build_explanation_result


st.set_page_config(
    page_title="VeriqURL",
    page_icon=str(LOGO_PATH),
    layout="wide",
)

#=========================
# Session State for URL Analysis History
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

def add_to_session_history(url: str, final_decision: dict):
    """
    Store recent analysis results in Streamlit session only.
    This does not write to a database or external file.
    """
    history_item = {
        "URL": url,
        "Final Label": final_decision["final_label"],
        "Final Risk": final_decision["final_risk"],
        "Average Phishing Probability": round(
            final_decision["average_phishing_probability"], 4
        ),
    }

    st.session_state.analysis_history.insert(0, history_item)

    # Keep only latest 5 records
    st.session_state.analysis_history = st.session_state.analysis_history[:5]

def display_session_history():
    """
    Display session-only URL analysis history.
    """
    if not st.session_state.analysis_history:
        st.info("No URL has been analysed in this session yet.")
        return

    history_df = pd.DataFrame(st.session_state.analysis_history)
    st.dataframe(history_df, hide_index=True, use_container_width=True)


# =========================
# Custom CSS
# Inspired by your HTML/CSS prototype
# =========================

st.markdown(
    """
    <style>
    :root {
        --safe: #16a34a;
        --warn: #d97706;
        --danger: #dc2626;
        --info: #2563eb;
        --text: #0f172a;
        --muted: #475569;
        --line: rgba(15, 23, 42, .12);
        --panel: #ffffff;
        --panel-2: #f8fafc;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: var(--text);
    }

    .subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 0;
    }

    .brand-box {
        padding: 1.1rem 1.25rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: linear-gradient(135deg, #eff6ff, #f0fdf4);
        margin-bottom: 1rem;
    }

    .card {
        padding: 1.1rem 1.25rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: var(--panel);
        box-shadow: 0 10px 26px rgba(15, 23, 42, .06);
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--panel-2);
        margin-bottom: 0.8rem;
    }

    .risk-low {
        color: white;
        background: var(--safe);
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-suspicious {
        color: white;
        background: var(--warn);
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-high {
        color: white;
        background: var(--danger);
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
    }

    .section-heading {
        font-weight: 800;
        font-size: 1.25rem;
        margin-top: 0.2rem;
        margin-bottom: 0.7rem;
    }

    .footer-note {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--line);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Cached model loading
# =========================

@st.cache_resource
def load_models_once():
    """
    Load RF and TCN models once.

    Streamlit reruns the script whenever the user interacts with the page.
    Caching avoids reloading the models every time.
    """
    rf_model = load_rf_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    char_to_idx = load_char_vocab()
    vocab_size = len(char_to_idx)

    tcn_model = load_tcn_model(vocab_size, device)

    return rf_model, tcn_model, char_to_idx, device


def risk_badge(risk_level: str) -> str:
    """
    Convert risk level into colored HTML badge.
    """
    if risk_level == "High Risk":
        return '<span class="risk-high">High Risk</span>'
    elif risk_level == "Suspicious":
        return '<span class="risk-suspicious">Suspicious</span>'
    return '<span class="risk-low">Low Risk</span>'

def parse_url_components(url: str) -> dict:
    """
    Parse URL into user-friendly components for awareness display.
    """
    clean_url = url.strip()

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", clean_url):
        parse_target = "http://" + clean_url
    else:
        parse_target = clean_url

    parsed = urlparse(parse_target)

    domain = parsed.netloc.lower()

    if "@" in domain:
        domain = domain.split("@")[-1]

    if ":" in domain:
        domain = domain.split(":")[0]

    labels = [part for part in domain.split(".") if part]

    if len(labels) >= 2:
        registered_domain = ".".join(labels[-2:])
        subdomain = ".".join(labels[:-2]) if len(labels) > 2 else "-"
    else:
        registered_domain = domain if domain else "-"
        subdomain = "-"

    return {
        "scheme": parsed.scheme if parsed.scheme else "-",
        "subdomain": subdomain,
        "domain": domain if domain else "-",
        "registered_domain": registered_domain,
        "path": parsed.path if parsed.path else "-",
        "query": parsed.query if parsed.query else "-",
    }


def highlight_url_indicators(url: str, indicators: dict) -> str:
    """
    Highlight detected suspicious words and brand words inside the URL.
    This is for visual awareness only and does not affect prediction.
    """
    highlighted_url = url

    words_to_highlight = []

    words_to_highlight.extend(indicators.get("matched_brand_words", []))
    words_to_highlight.extend(indicators.get("matched_suspicious_words", []))

    # Longer words first to avoid partial replacement issues
    words_to_highlight = sorted(set(words_to_highlight), key=len, reverse=True)

    for word in words_to_highlight:
        if not word:
            continue

        pattern = re.compile(re.escape(word), re.IGNORECASE)

        highlighted_url = pattern.sub(
            lambda m: (
                f"<mark style='background-color:#fde68a; "
                f"padding:2px 4px; border-radius:4px;'>"
                f"{m.group(0)}</mark>"
            ),
            highlighted_url
        )

    return highlighted_url


def display_url_component_visualizer(url: str, indicators: dict):
    """
    Display URL components and highlight suspicious / brand-related tokens.
    """
    components = parse_url_components(url)

    with st.container(border=True):
        st.subheader("URL Component Visualizer")

        st.caption(
            "This section helps users understand where the scheme, subdomain, registered domain, path, and query appear in the URL. "
            "Highlighted words are awareness indicators only and do not directly change the model prediction."
        )

        component_df = pd.DataFrame(
            [
                {"Component": "Scheme", "Value": components["scheme"]},
                {"Component": "Subdomain", "Value": components["subdomain"]},
                {"Component": "Full domain / host", "Value": components["domain"]},
                {"Component": "Approx. registered domain", "Value": components["registered_domain"]},
                {"Component": "Path", "Value": components["path"]},
                {"Component": "Query", "Value": components["query"]},
            ]
        )

        st.dataframe(component_df, hide_index=True, use_container_width=True)

        st.markdown("#### Highlighted URL")
        highlighted_url = highlight_url_indicators(url, indicators)

        st.markdown(
            f"""
            <div style="
                border:1px solid rgba(15,23,42,.12);
                border-radius:12px;
                padding:0.85rem;
                background:#f8fafc;
                overflow-wrap:anywhere;
                font-family:monospace;
                font-size:0.95rem;
            ">
                {highlighted_url}
            </div>
            """,
            unsafe_allow_html=True,
        )

def display_model_result(title: str, result: dict):
    """
    Display one model's prediction result.
    """
    with st.container(border=True):
        st.subheader(title)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Label", result["predicted_label"])

        with col2:
            st.metric("Phishing Probability", f"{result['phishing_probability']:.4f}")

        with col3:
            st.markdown("**Risk Level**")
            st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)

        st.progress(float(result["phishing_probability"]))


def display_combined_result(final_decision: dict):
    """
    Display combined RF + TCN decision.
    """
    with st.container(border=True):
        st.subheader("Combined Decision")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Models Agree", final_decision["models_agree"])

        with col2:
            st.metric(
                "Average Phishing Probability",
                f"{final_decision['average_phishing_probability']:.4f}"
            )

        with col3:
            st.metric("Final Label", final_decision["final_label"])

        with col4:
            st.markdown("**Final Risk Level**")
            st.markdown(risk_badge(final_decision["final_risk"]), unsafe_allow_html=True)


def display_explanation(explanation: dict, risk_level: str):
    """
    Display shared explanation and awareness guidance.
    """
    with st.container(border=True):
        st.subheader("Why It Was Flagged")

        for i, reason in enumerate(explanation["reasons"], start=1):
            st.write(f"{i}. {reason}")

        st.markdown("### Awareness Guidance")

        for i, item in enumerate(explanation["awareness_guidance"], start=1):
            message = f"{i}. {item}"
            if risk_level == "High Risk":
                st.error(message)
            elif risk_level == "Suspicious":
                st.warning(message)
            else:
                st.info(message)


def display_lightweight_comparison():
    """
    Display RF vs TCN lightweight comparison table if available.
    """
    comparison_path = ROOT_DIR / "results" / "model_comparison.csv"

    if not comparison_path.exists():
        st.warning("model_comparison.csv not found. Run src/compare_models.py first.")
        return

    comparison_df = pd.read_csv(comparison_path)

    display_cols = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "training_time_s",
        "testing_time_s",
        "per_sample_latency_ms",
        "tes",
        "ies",
        "rtde",
        "model_size_mb",
        "num_parameters",
    ]

    available_cols = [col for col in display_cols if col in comparison_df.columns]

    st.dataframe(
        comparison_df[available_cols],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "TES = Accuracy / Training Time, "
        "IES = Accuracy / Testing Time, "
        "RTDE = Accuracy / Per-sample Latency."
    )


# =========================
# Main App
# =========================

if LOGO_PATH.exists():
    logo_base64 = image_to_base64(LOGO_PATH)

    st.markdown(
        f"""
        <div class="brand-box" style="
            display: flex;
            align-items: center;
            gap: 1.4rem;
        ">
            <img src="data:image/png;base64,{logo_base64}" 
                 style="width: 95px; height: 95px; object-fit: contain;">
            <div>
                <div class="main-title">VeriqURL</div>
                <div class="subtitle">
                    Lightweight URL-based phishing detection and awareness tool using Random Forest and TCN.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="brand-box">
            <div class="main-title">🛡️ VeriqURL</div>
            <div class="subtitle">
                Lightweight URL-based phishing detection and awareness tool using Random Forest and TCN.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_analyse, tab_metrics, tab_about = st.tabs(
    ["Analyse URL", "Model & Lightweight Metrics", "About"]
)


with tab_analyse:
    st.markdown("## Analyse a URL")

    st.write(
        "Paste a URL below. The system will run both Random Forest and TCN, "
        "then generate a shared explanation and evidence-based awareness guidance."
    )

    default_url = "https://www.google.com"
    url = st.text_input("URL to analyse", value=default_url)

    analyse_clicked = st.button("Analyze URL", type="primary")

    if analyse_clicked:
        if not url.strip():
            st.error("Please enter a URL.")

        elif url.count("http://") + url.count("https://") > 1:
            st.error("It looks like more than one URL was pasted. Please enter only one URL.")

        else:
            clean_url = url.strip()

            with st.spinner("Loading models and analysing URL..."):
                rf_model, tcn_model, char_to_idx, device = load_models_once()

                rf_result = predict_rf(clean_url, rf_model)
                tcn_result = predict_tcn(clean_url, tcn_model, char_to_idx, device)

                final_decision = get_final_decision(rf_result, tcn_result)

                shared_explanation = build_explanation_result(
                    clean_url,
                    max(
                        rf_result["phishing_probability"],
                        tcn_result["phishing_probability"]
                    )
                )

            st.markdown("---")

            add_to_session_history(clean_url, final_decision)

            display_model_result("Random Forest Result", rf_result)
            display_model_result("TCN Result", tcn_result)
            display_combined_result(final_decision)
            display_explanation(shared_explanation, final_decision["final_risk"])
            display_url_component_visualizer(clean_url, shared_explanation["indicators"])

            with st.expander("Show RF Model Features"):
                st.json(rf_result["features"])

            with st.expander("Show TCN Sequence Info"):
                st.write("Fixed sequence length:", tcn_result["encoded_sequence_length"])
                st.write("Non-padding character count:", tcn_result["non_padding_length"])

            with st.expander("Show Explanation Indicators"):
                st.json(shared_explanation["indicators"])

    st.markdown("---")
    st.subheader("Session-only Analysis History")
    display_session_history()


with tab_metrics:
    st.markdown("## Model & Lightweight Metrics")

    st.write(
        "This section compares Random Forest and TCN using detection performance "
        "and lightweight efficiency metrics."
    )

    display_lightweight_comparison()

    st.markdown(
        """
        ### Interpretation

        Random Forest is the more lightweight model in terms of training time,
        testing time, and per-sample latency. TCN achieves slightly stronger
        detection performance, especially recall and F1-score, while still having
        acceptable real-time inference latency.
        """
    )


with tab_about:
    st.markdown("## About VeriqURL")

    st.markdown(
        """
        VeriqURL is a lightweight URL-based phishing detection prototype.

        The system uses two models:

        - **Random Forest**: a fast baseline model using handcrafted lexical URL features.
        - **TCN**: a character-level Temporal Convolutional Network that learns URL sequence patterns.

        The system also includes a shared explanation layer that generates:

        - risk level,
        - human-readable reasons,
        - evidence-based awareness guidance.

        Additional awareness features include:

        - URL component visualizer,
        - highlighted suspicious or brand-related URL tokens,
        - session-only analysis history.

        The prototype does not visit the webpage and does not collect usernames,
        passwords, banking details, or personal data. The session-only history is
        not saved to a database.
        """
    )

st.markdown(
    """
    <div class="footer-note">
        VeriqURL prototype • URL-only phishing detection • No webpage content is downloaded.
    </div>
    """,
    unsafe_allow_html=True,
)