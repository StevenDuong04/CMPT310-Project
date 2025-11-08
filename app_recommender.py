# streamlit run app_recommender.py
from __future__ import annotations

from typing import List

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
            1. Upload a CSV or Excel file containing user profiles.
            2. Click **Submit** to generate placeholder recommendations.
            3. Review the mock Top 3 list, then reset to try another file.
            """
        )


def _safe_rerun() -> None:
    """Call st.rerun or st.experimental_rerun if available."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def render_onboarding_modal() -> None:
    """Show a blocking onboarding UI until the user acknowledges it."""
    if st.session_state.get("onboarding_seen", False):
        return

    # Newer Streamlit supports st.modal
    if hasattr(st, "modal"):
        with st.modal("Welcome to Career Path Recommender", key="onboarding_modal"):
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
                1. Click **Upload your dataset** and select a CSV or Excel file.
                2. Press **Submit** to see the (mock) Top 3 recommendations.
                3. Use **Run another file** to reset and try a different dataset.
                
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
        st.write("👋 **Hi!** Thanks for trying our app.")
        st.markdown(
            """
            **How to use this app**
            1. Click **Upload your dataset** and select a CSV or Excel file.\
            2. Press **Submit** to see the (mock) Top 3 recommendations.\
            3. Use **Run another file** to reset and try a different dataset.

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


def get_top3_recommendations(_uploaded_file) -> List[str]:
    # TODO: Implement the actual recommendation engine using the uploaded data.
    return ["Software Engineer", "Data Scientist", "Product Manager"]


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
