"""
Tab 3: Test Model

Handles testing the fine-tuned model with custom prompts.
"""

import json
from pathlib import Path
import streamlit as st

from lib.config import MODEL_INFO_FILE, FINETUNING_BASE_MODEL

from .shared import generate_email, get_openai_client


def render():
    """Render the Test Model tab."""
    st.header("Test Your Model")
    st.markdown("Try out your fine-tuned email writing model.")

    # Check if model exists
    if not Path(MODEL_INFO_FILE).exists():
        st.warning("No fine-tuned model found. Please complete fine-tuning first in the 'Fine-Tune Model' tab.")
        st.stop()

    # Load model info
    try:
        with open(MODEL_INFO_FILE, 'r') as f:
            model_info = json.load(f)

        model_id = model_info.get("model_id")
        base_model = model_info.get("base_model", FINETUNING_BASE_MODEL)

        st.success(f"Model loaded: `{model_id}`")

        # Quick examples
        st.subheader("Quick Test Examples")

        examples = [
            "Write a brief email asking a colleague if they're free for coffee next week.",
            "Write an email thanking someone for their help on a project.",
            "Write a short message declining a meeting invitation politely.",
        ]

        selected_example = st.selectbox("Choose an example prompt:", [""] + examples)

        # Custom prompt
        st.subheader("Or Enter Your Own Prompt")
        prompt = st.text_area(
            "Prompt",
            value=selected_example if selected_example else "",
            placeholder="e.g., Write an email asking for a project update...",
            height=100
        )

        # Comparison option
        compare_with_base = st.checkbox("Compare with base model", value=False)

        # Generate button
        if st.button("Generate Email", type="primary", disabled=not prompt):
            try:
                client = get_openai_client()

                if compare_with_base:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Fine-Tuned Model")
                        with st.spinner("Generating..."):
                            fine_tuned_response = generate_email(client, model_id, prompt)
                        st.markdown("---")
                        st.write(fine_tuned_response)

                    with col2:
                        st.markdown("### Base Model")
                        with st.spinner("Generating..."):
                            base_response = generate_email(client, base_model, prompt)
                        st.markdown("---")
                        st.write(base_response)
                else:
                    with st.spinner("Generating email..."):
                        response = generate_email(client, model_id, prompt)

                    st.markdown("### Generated Email")
                    st.markdown("---")
                    st.write(response)
                    st.markdown("---")

            except Exception as e:
                st.error(f"Error generating email: {e}")

    except Exception as e:
        st.error(f"Error loading model info: {e}")
