#!/usr/bin/env python3
"""
Email Writer - Web UI

A Streamlit-based web interface for fine-tuning OpenAI models to write emails
in your personal style.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
from dotenv import load_dotenv

from ui.shared import apply_custom_css, get_api_key
from ui import prepare_data_ui, finetune_ui, test_model_ui

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Email Writer",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS
apply_custom_css()


def main():
    """Main app entry point."""
    # Header
    st.markdown('<div class="main-header">📧 Email Writer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fine-tune AI models to write emails in your personal style</div>', unsafe_allow_html=True)

    # API Key Configuration
    api_key = get_api_key()

    if not api_key:
        st.warning("⚠️ OpenAI API key required to continue")

        with st.expander("🔑 Configure API Key", expanded=True):
            st.markdown("""
            Enter your OpenAI API key to get started. You can get one from the
            [OpenAI Platform](https://platform.openai.com/api-keys).

            Your API key will be stored securely in your browser session and is never saved to disk.
            """)

            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Get your API key from https://platform.openai.com/api-keys"
            )

            if st.button("Save API Key", type="primary"):
                if not api_key_input:
                    st.error("Please enter an API key")
                elif api_key_input.startswith("sk-"):
                    st.session_state.openai_api_key = api_key_input
                    st.success("✅ API key saved! The page will refresh.")
                    st.rerun()
                else:
                    st.error("Invalid API key format. OpenAI API keys start with 'sk-'")

        st.info("💡 **Tip:** You can also set the `OPENAI_API_KEY` environment variable in a `.env` file instead.")
        st.stop()

    # Show API key status (masked)
    with st.sidebar:
        st.success("🔑 API Key Configured")
        if st.button("Clear API Key"):
            if 'openai_api_key' in st.session_state:
                del st.session_state.openai_api_key
            st.rerun()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["1️⃣ Prepare Data", "2️⃣ Fine-Tune Model", "3️⃣ Test Model"])

    # Render each tab
    with tab1:
        prepare_data_ui.render()

    with tab2:
        finetune_ui.render()

    with tab3:
        test_model_ui.render()


if __name__ == "__main__":
    main()
