"""
Tab 2: Fine-Tune Model

Handles creating and monitoring fine-tuning jobs.
"""

import json
from pathlib import Path
import streamlit as st

from lib.config import (
    TRAINING_FILE,
    VALIDATION_FILE,
    MODEL_INFO_FILE,
    FINETUNING_BASE_MODEL,
    FINE_TUNING_SUFFIX,
    FINE_TUNING_HYPERPARAMETERS
)

from .shared import upload_file_to_openai, create_fine_tuning_job, get_openai_client


def render():
    """Render the Fine-Tune Model tab."""
    st.header("Fine-Tune Model")
    st.markdown("Create a personalized email writing model using your data.")

    # Check if training data exists
    if not Path(TRAINING_FILE).exists():
        st.warning("No training data found. Please prepare data first in the 'Prepare Data' tab.")
        st.stop()

    # Display dataset info
    st.success("Training data ready!")

    # Count examples
    train_count = sum(1 for _ in open(TRAINING_FILE))
    val_count = sum(1 for _ in open(VALIDATION_FILE)) if Path(VALIDATION_FILE).exists() else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Examples", train_count)
    with col2:
        st.metric("Validation Examples", val_count)
    with col3:
        st.metric("Base Model", FINETUNING_BASE_MODEL.split('-')[0])

    st.markdown("---")

    # Initialize session state
    if 'finetuning_job_id' not in st.session_state:
        st.session_state.finetuning_job_id = None
    if 'finetuning_status' not in st.session_state:
        st.session_state.finetuning_status = None

    # Start fine-tuning button
    if not st.session_state.finetuning_job_id:
        if st.button("Start Fine-Tuning", type="primary"):
            try:
                client = get_openai_client()

                with st.spinner("Uploading files to OpenAI..."):
                    training_file_id = upload_file_to_openai(client, TRAINING_FILE, "fine-tune")
                    validation_file_id = None
                    if Path(VALIDATION_FILE).exists():
                        validation_file_id = upload_file_to_openai(client, VALIDATION_FILE, "fine-tune")
                    st.success("✓ Files uploaded")

                with st.spinner("Creating fine-tuning job..."):
                    job_id = create_fine_tuning_job(
                        client,
                        training_file_id,
                        validation_file_id,
                        base_model=FINETUNING_BASE_MODEL,
                        suffix=FINE_TUNING_SUFFIX,
                        hyperparameters=FINE_TUNING_HYPERPARAMETERS
                    )
                    st.session_state.finetuning_job_id = job_id
                    st.success(f"✓ Fine-tuning job created: {job_id}")
                    st.rerun()

            except Exception as e:
                st.error(f"Error starting fine-tuning: {e}")

    # Monitor existing job
    if st.session_state.finetuning_job_id:
        st.info(f"Job ID: {st.session_state.finetuning_job_id}")

        # Manual refresh button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Refresh Status"):
                st.rerun()

        try:
            client = get_openai_client()
            job = client.fine_tuning.jobs.retrieve(st.session_state.finetuning_job_id)
            status = job.status

            # Display status
            status_colors = {
                "validating_files": "🔵",
                "queued": "🟡",
                "running": "🟠",
                "succeeded": "🟢",
                "failed": "🔴",
                "cancelled": "⚫"
            }

            st.markdown(f"### {status_colors.get(status, '⚪')} Status: {status}")

            if job.trained_tokens:
                st.metric("Trained Tokens", f"{job.trained_tokens:,}")

            # Handle completion
            if status == "succeeded":
                st.success("✅ Fine-tuning completed successfully!")
                st.code(job.fine_tuned_model, language=None)

                # Save model info
                model_info = {
                    "job_id": st.session_state.finetuning_job_id,
                    "model_id": job.fine_tuned_model,
                    "base_model": FINETUNING_BASE_MODEL,
                    "trained_tokens": job.trained_tokens,
                    "status": status
                }

                with open(MODEL_INFO_FILE, 'w') as f:
                    json.dump(model_info, f, indent=2)

                st.success(f"Model info saved to {MODEL_INFO_FILE}")
                st.info("Go to the 'Test Model' tab to try your fine-tuned model!")

                # Reset session state
                if st.button("Reset (start new fine-tuning)"):
                    st.session_state.finetuning_job_id = None
                    st.session_state.finetuning_status = None
                    st.rerun()

            elif status == "failed":
                st.error("❌ Fine-tuning failed!")
                if job.error:
                    st.error(f"Error: {job.error}")

                if st.button("Reset"):
                    st.session_state.finetuning_job_id = None
                    st.session_state.finetuning_status = None
                    st.rerun()

            elif status == "cancelled":
                st.warning("⚠️ Fine-tuning was cancelled")

                if st.button("Reset"):
                    st.session_state.finetuning_job_id = None
                    st.session_state.finetuning_status = None
                    st.rerun()

            else:
                st.info("Fine-tuning in progress. This may take 20-60 minutes. Refresh this page to check status.")
                st.info("💡 You can close this page and come back later - the job will continue running.")

        except Exception as e:
            st.error(f"Error monitoring job: {e}")
