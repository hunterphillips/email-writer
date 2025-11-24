"""
Tab 1: Prepare Training Data

Handles uploading .mbox files and creating training datasets.
"""

from pathlib import Path
import streamlit as st

from lib.email_cleaner import process_mbox
from lib.prompt_enhancer import enhance_generic_prompts
from lib.config import TRAINING_FILE, VALIDATION_FILE, MIN_TRAINING_EXAMPLES

from .shared import estimate_cost, split_dataset, write_jsonl, get_openai_client


def render():
    """Render the Prepare Data tab."""
    st.header("Prepare Training Data")
    st.markdown("Upload your Gmail .mbox export to create training data.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload .mbox file",
            type=['mbox'],
            help="Export your Sent folder from Gmail using Google Takeout"
        )

        user_email = st.text_input(
            "Your email address",
            placeholder="you@gmail.com",
            help="This helps identify which emails are yours in the .mbox file"
        )

    with col2:
        st.info("""
        **How to get your .mbox file:**

        1. Go to [Google Takeout](https://takeout.google.com/)
        2. Select only "Mail" → "Sent"
        3. Download and extract
        4. Upload the Sent.mbox file here
        """)

    if st.button("Process Data", type="primary", disabled=not (uploaded_file and user_email)):
        with st.spinner("Processing emails..."):
            try:
                # Save uploaded file temporarily
                temp_path = Path("temp_uploaded.mbox")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Step 1: Extract and clean emails
                st.info("Step 1/4: Extracting and cleaning emails...")
                dataset = process_mbox(str(temp_path), user_email)
                st.success(f"✓ Found {len(dataset)} clean email examples")

                # Check minimum requirements
                if len(dataset) < MIN_TRAINING_EXAMPLES:
                    st.error(f"Only {len(dataset)} examples found. OpenAI requires at least {MIN_TRAINING_EXAMPLES} examples.")
                    temp_path.unlink()
                    st.stop()

                # Step 2: Enhance generic prompts
                st.info("Step 2/4: Enhancing generic prompts...")
                try:
                    client = get_openai_client()
                    enhanced_dataset, api_calls = enhance_generic_prompts(client, dataset, verbose=False)
                    st.success(f"✓ Enhanced prompts using {api_calls} API calls")
                except Exception as e:
                    st.warning(f"Prompt enhancement failed: {e}. Continuing with original prompts...")
                    enhanced_dataset = dataset
                    api_calls = 0

                # Step 3: Split dataset
                st.info("Step 3/4: Splitting into training and validation sets...")
                train_examples, val_examples = split_dataset(enhanced_dataset)
                st.success(f"✓ Training: {len(train_examples)} examples, Validation: {len(val_examples)} examples")

                # Step 4: Write files
                st.info("Step 4/4: Writing output files...")
                write_jsonl(train_examples, TRAINING_FILE)
                write_jsonl(val_examples, VALIDATION_FILE)
                st.success(f"✓ Created {TRAINING_FILE} and {VALIDATION_FILE}")

                # Cost estimation
                costs = estimate_cost(train_examples, api_calls)

                st.markdown("---")
                st.subheader("Cost Estimate")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tokens", f"{costs['total_tokens']:,}")
                with col2:
                    st.metric("Enhancement", f"${costs['enhancement_cost_usd']:.3f}")
                with col3:
                    st.metric("Fine-tuning", f"${costs['training_cost_usd']:.3f}")
                with col4:
                    st.metric("Total Cost", f"${costs['total_cost_usd']:.2f}")

                st.success("✅ Data preparation complete! Go to the 'Fine-Tune Model' tab to continue.")

                # Clean up temp file
                temp_path.unlink()

            except Exception as e:
                st.error(f"Error processing data: {e}")
                if temp_path.exists():
                    temp_path.unlink()
