# streamlit run app_recommender.py
from __future__ import annotations

from typing import List
from typing import Any, Callable, ContextManager, Optional, cast

import pandas as pd
import streamlit as st


def init_state() -> None:
    """Initialize the Streamlit session state once per session."""
    defaults = {
        "uploaded_file": None,
        "submitted": False,
        "recommendations": [],
        "onboarding_seen": False,
        "uploader_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    st.set_page_config(page_title="Career Path Recommender", layout="wide")
    st.title("Career Path Recommender")
    st.caption("Upload a dataset to preview mock career path recommendations.")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("How it works")
        st.markdown(
            """
            1. **Click** *Upload your dataset* and select a CSV or Excel file containing your grades.  
            2. **Press** *Submit* to see the (mock) Top 3 recommendations.  
            3. **Use** *Run another file* to reset and try a different dataset.  
            """
        )


def _safe_rerun() -> None:
    """Call st.rerun or st.experimental_rerun if available (without static attribute access)."""
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    exp_rerun = getattr(st, "experimental_rerun", None)
    if callable(exp_rerun):
        exp_rerun()

def render_onboarding_modal() -> None:
    """Show a blocking onboarding UI until the user acknowledges it."""
    if st.session_state.get("onboarding_seen", False):
        return

    # Newer Streamlit supports st.modal; type-cast so Pylance sees a context manager
    ModalType = Callable[..., ContextManager[Any]]
    modal = cast(Optional[ModalType], getattr(st, "modal", None))
    if modal is not None:
        cm: ContextManager[Any] = modal("Welcome to Career Path Recommender", key="onboarding_modal")
        with cm:
            st.markdown(
                """
                <style>
                    [data-testid="stModal"] * { color: #000000 !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.write("👋 **Hi!** Thanks for trying our app.")
            st.markdown(
                """
                **How to use this app**

                1. **Click** *Upload your dataset* and select a CSV or Excel file containing your grades.  
                2. **Press** *Submit* to see the (mock) Top 3 recommendations.  
                3. **Use** *Run another file* to reset and try a different dataset.  

                _Note: Recommendations are placeholders until the ML pipeline is connected._
                """
            )
            if st.button("I understand — let's start", key="onboard_ack_modal", use_container_width=True):
                st.session_state.onboarding_seen = True
                _safe_rerun()
        # Block the rest of the page until acknowledged
        st.stop()
        return

    # Fallback for older Streamlit versions without st.modal: use a blocking form
    st.markdown("## Welcome to Career Path Recommender")
    with st.form(key="onboard_form"):
        st.write("**Hi!** Thanks for trying our app.")
        st.markdown(
            """
            **How to use this app**

            1. **Click** *Upload your dataset* and select a CSV or Excel file containing your grades.  
            2. **Press** *Submit* to see the (mock) Top 3 recommendations.  
            3. **Use** *Run another file* to reset and try a different dataset.  

            _Note: Recommendations are placeholders until the ML pipeline is connected._
            """
        )
        acknowledged = st.form_submit_button("I understand — let's start")
    if acknowledged:
        st.session_state.onboarding_seen = True
        _safe_rerun()
    # Block the rest of the page until acknowledged
    st.stop()


def render_upload_form() -> None:
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key=f"dataset_uploader_{st.session_state.uploader_nonce}",
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    submit_disabled = st.session_state.uploaded_file is None
    submitted = st.button("Submit", disabled=submit_disabled)

    if submitted and st.session_state.uploaded_file is not None:
        # TODO: Replace mock inference with real preprocessing + ML pipeline.
        st.session_state.recommendations = get_top3_recommendations(
            st.session_state.uploaded_file
        )
        st.session_state.submitted = True


DEFAULT_RECOMMENDATIONS: List[str] = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
]

CAREER_KEYWORDS = {
    "Software Engineer": ("math", "calculus", "cs", "program", "logic", "computer"),
    "Data Scientist": ("data", "stat", "analytics", "machine", "ml", "ai"),
    "Product Manager": ("business", "manage", "communication", "english", "marketing"),
    "Mechanical Engineer": ("physics", "mechanic", "engineering", "manufacturing"),
    "Biotechnologist": ("bio", "life", "medical", "health"),
    "Chemical Engineer": ("chem", "chemical", "materials"),
    "Financial Analyst": ("finance", "econom", "account", "market"),
    "UX Designer": ("design", "creative", "art", "visual"),
}


def get_top3_recommendations(_uploaded_file) -> List[str]:
    if _uploaded_file is None:
        return DEFAULT_RECOMMENDATIONS.copy()

    filename = getattr(_uploaded_file, "name", "").lower()
    preferred_reader = pd.read_csv if filename.endswith(".csv") else pd.read_excel
    dataframe = None
    try:
        dataframe = preferred_reader(_uploaded_file)
    except Exception:
        try:
            _uploaded_file.seek(0)
            fallback_reader = pd.read_excel if preferred_reader is pd.read_csv else pd.read_csv
            dataframe = fallback_reader(_uploaded_file)
        except Exception:
            return DEFAULT_RECOMMENDATIONS.copy()
    finally:
        try:
            _uploaded_file.seek(0)
        except Exception:
            pass

    numeric_data = dataframe.select_dtypes(include="number")
    if numeric_data.empty:
        return DEFAULT_RECOMMENDATIONS.copy()

    col_means = numeric_data.mean(axis=0, numeric_only=True)
    if col_means.empty:
        return DEFAULT_RECOMMENDATIONS.copy()

    career_scores = {career: 0.0 for career in CAREER_KEYWORDS}
    unmatched_strength = 0.0

    for col_name, mean_value in col_means.items():
        col_key = col_name.lower()
        adjusted_value = float(mean_value)
        if "absent" in col_key or "absence" in col_key:
            adjusted_value = max(0.0, 100.0 - adjusted_value)
        matched = False
        for career, keywords in CAREER_KEYWORDS.items():
            if any(keyword in col_key for keyword in keywords):
                career_scores[career] += adjusted_value
                matched = True
        if not matched:
            unmatched_strength += adjusted_value

    if unmatched_strength > 0.0:
        spread = (unmatched_strength * 0.1) / max(len(career_scores), 1)
        for career in career_scores:
            career_scores[career] += spread

    ranked = sorted(
        (item for item in career_scores.items() if item[1] > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )

    recommendations = [career for career, _ in ranked[:3]]
    if len(recommendations) < 3:
        for career in DEFAULT_RECOMMENDATIONS:
            if career not in recommendations:
                recommendations.append(career)
            if len(recommendations) == 3:
                break

    return recommendations


def render_results() -> None:
    if not st.session_state.submitted:
        return

    st.markdown("---")
    st.subheader("Top 3 Recommendations")
    if not st.session_state.recommendations:
        st.info("No recommendations available yet.")
    else:
        for rec in st.session_state.recommendations:
            st.markdown(f"- {rec}")

    if st.button("Run another file"):
        st.session_state.uploaded_file = None
        st.session_state.submitted = False
        st.session_state.recommendations = []
        # Bump nonce to force a fresh uploader widget key on next render
        st.session_state.uploader_nonce += 1
        _safe_rerun()


def main() -> None:
    init_state()
    render_header()
    render_onboarding_modal()
    render_sidebar()
    render_upload_form()
    render_results()


if __name__ == "__main__":
    main()
