# Authors: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
# Project: CMPT 310 Final Project - Career Path Prediction
# Date: Nov 10, 2025
# Description: This file contains the code for the user interface using Streamlit.

# To run the app, use the command:
# streamlit run app_recommender.py

# Import necessary libraries
from __future__ import annotations
from model import predict_career_path
from typing import List
import pandas as pd
import streamlit as st
import tempfile

# Initialize Streamlit session state
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

# Render the app header
def render_header() -> None:
    st.set_page_config(page_title="FindMyPath", layout="wide")
    st.title("FindMyPath: Career Path Recommender")
    st.caption("Upload a transcript dataset to view career path recommendations.")

# Render the sidebar with instructions
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

# Safely rerun the app
def _safe_rerun() -> None:
    """Call st.rerun or st.experimental_rerun if available."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# Render the onboarding modal
def render_onboarding_modal() -> None:
    """Show a blocking onboarding UI until the user acknowledges it."""
    if st.session_state.get("onboarding_seen", False):
        return

    # Newer Streamlit supports st.modal
    if hasattr(st, "modal"):
        with st.modal("Welcome to FindMyPath", key="onboarding_modal"):
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
                1. Click "Upload your dataset" and select a CSV or Excel file.
                2. Press "Submit" to see the Top 3 recommendations.
                3. Use "Run another file" to reset and try a different dataset.
                """
            )
            if st.button("I understand — let's start", key="onboard_ack_modal", use_container_width=True):
                st.session_state.onboarding_seen = True
                _safe_rerun()
        # Block the rest of the page until acknowledged
        st.stop()
        return

    # Fallback for older Streamlit versions without st.modal: use a blocking form
    st.markdown("## Welcome to FindMyPath")
    with st.form(key="onboard_form"):
        st.write("👋 **Hi!** Thanks for trying our app.")
        st.markdown(
            """
            **How to use this app**
            1. Click "Upload your dataset" and select a CSV or Excel file.
            2. Press "Submit" to see the Top 3 recommendations.
            3. Use "Run another file" to reset and try a different dataset.
            """
        )
        acknowledged = st.form_submit_button("I understand — let's start")
    if acknowledged:
        st.session_state.onboarding_seen = True
        _safe_rerun()
    # Block the rest of the page until acknowledged
    st.stop()

# Render alert for parse errors
def render_parse_error_alert() -> None:
    st.markdown(
        """
        <style>
        .notifications-container {
          width: 100%;
          max-width: 480px;
          font-size: 0.875rem;
          line-height: 1.25rem;
          display: flex;
          flex-direction: column;
          gap: 1rem;
          margin-top: 12px;
        }

        .flex {
          display: flex;
        }

        .flex-shrink-0 {
          flex-shrink: 0;
        }

        .alert {
          background-color: rgb(254, 252, 232);
          border-left-width: 4px;
          border-color: rgb(250, 204, 21);
          border-radius: 0.375rem;
          padding: 1rem;
          border-style: solid;
        }

        .alert-svg {
          height: 1.25rem;
          width: 1.25rem;
          color: rgb(250, 204, 21);
        }

        .alert-prompt-wrap {
          margin-left: 0.75rem;
          color: rgb(202, 138, 4);
        }

        .alert-prompt-link {
          font-weight: 500;
          color: rgb(141, 56, 0);
          text-decoration: underline;
          cursor: pointer;
        }

        .alert-prompt-link:hover {
          color: rgb(202, 138, 4);
        }
        </style>

        <div class="notifications-container">
          <div class="alert">
            <div class="flex">
              <div class="flex-shrink-0">
                <svg aria-hidden="true" fill="currentColor" viewBox="0 0 20 20"
                     xmlns="http://www.w3.org/2000/svg" class="alert-svg">
                  <path clip-rule="evenodd"
                        d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0
                             1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                        fill-rule="evenodd"></path>
                </svg>
              </div>
              <div class="alert-prompt-wrap">
                <p class="text-sm">
                  Sorry, but we couldn't read your file. Please remove the uploaded file and try again with a
                  <a class="alert-prompt-link" href="https://support.collegekickstart.com/hc/en-us/articles/11386233650445-How-do-I-export-a-file-as-a-CSV-in-UTF-8#:~:text=In%20order%20to%20read%20special,that%20can%20support%20many%20languages." target="_blank" rel="noopener noreferrer">
                    UTF-8 encoded CSV file
                  </a>.
                </p>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render the file upload form
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

# Get top 3 career path recommendations (mock implementation)
def get_top3_recommendations(_uploaded_file) -> List[str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(_uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        top3 = predict_career_path(tmp_path)
    except ValueError as exc:
        if str(exc) == "PARSE_ERROR":
            st.session_state.uploaded_file = None
            st.session_state.submitted = False
            st.session_state.recommendations = []
            st.session_state.uploader_nonce += 1
            render_parse_error_alert()
            return []
        raise
    return top3

# Render the results section
def render_results() -> None:
    if not st.session_state.submitted:
        return

    st.markdown("---")
    st.subheader("Top 3 Recommendations")
    if not st.session_state.recommendations:
        st.info("No recommendations available yet.")
    else:
        for career, prob in st.session_state.recommendations:
            st.write(f"**{career}** - {prob*100:.1f}%")
            st.progress(prob)
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
