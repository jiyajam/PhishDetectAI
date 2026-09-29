
# ============================================================
# PhishDetectAI
# Streamlit application
#
# Pages:
# 1. Email Detection
# 2. Model Evaluation
#
# Models:
# - Fine-Tuned Qwen
# - RAG + Fine-Tuned Qwen
# - TF-IDF + Logistic Regression
# ============================================================

import os
import torch
import streamlit as st
import chromadb

import unsloth
from unsloth import FastLanguageModel

from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PhishDetectAI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/content/drive/MyDrive/PhishDetectAI"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "qwen_phishing_unsloth_correct"
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "rag_chroma_correct"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = (
    "You are a cybersecurity expert specializing in phishing detection. "
    "Analyze the provided email, message, or URL. "
    "Determine whether it is a phishing attempt or a legitimate communication. "
    "Respond with only one word: 'phishing' or 'legitimate'."
)


# ============================================================
# LOAD QWEN MODEL
# ============================================================

@st.cache_resource
def load_qwen_model():

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=True
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD CHROMADB COLLECTION
# ============================================================

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name="phishing_emails"
    )

    return collection


# ============================================================
# LOAD MODELS
# ============================================================

with st.spinner("Loading AI models..."):

    model, tokenizer = load_qwen_model()

    embedding_model = load_embedding_model()

    collection = load_collection()


# ============================================================
# HELPER — EXTRACT LABEL
# ============================================================

def extract_label(generated_text):

    generated_text = generated_text.strip().lower()

    if "phishing" in generated_text:
        return "phishing"

    elif "legitimate" in generated_text:
        return "legitimate"

    else:
        return "invalid"


# ============================================================
# DIRECT FINE-TUNED QWEN
# ============================================================

def predict_with_qwen(email_text):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": email_text
        }
    ]

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_input,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = (
        outputs[0][inputs.input_ids.shape[1]:]
    )

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    prediction = extract_label(
        generated_text
    )

    return prediction


# ============================================================
# RAG + FINE-TUNED QWEN
# ============================================================

def predict_with_rag_qwen(
    email_text,
    n_results=3
):

    # --------------------------------------------------------
    # STEP 1 — CREATE EMAIL EMBEDDING
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        email_text,
        show_progress_bar=False
    ).tolist()

    # --------------------------------------------------------
    # STEP 2 — RETRIEVE TOP SIMILAR EMAILS
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    # --------------------------------------------------------
    # STEP 3 — CREATE EVIDENCE
    # --------------------------------------------------------

    evidence_text = ""

    retrieved_count = len(
        results["documents"][0]
    )

    for i in range(retrieved_count):

        retrieved_email = (
            results["documents"][0][i]
        )

        retrieved_label = (
            results["metadatas"][0][i]["label"]
        )

        email_excerpt = retrieved_email[:300]

        evidence_text += (
            f"\n--- Evidence {i + 1} ---\n"
            f"Label: {retrieved_label}\n"
            f"Email excerpt: {email_excerpt}\n"
        )

    # --------------------------------------------------------
    # STEP 4 — CREATE RAG PROMPT
    # --------------------------------------------------------

    rag_user_content = f"""
Email to classify:

{email_text[:2500]}

Retrieved similar email evidence:

{evidence_text}

Classify the original email.

Respond with only one word: 'phishing' or 'legitimate'.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": rag_user_content
        }
    ]

    # --------------------------------------------------------
    # STEP 5 — APPLY CHAT TEMPLATE
    # --------------------------------------------------------

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # --------------------------------------------------------
    # STEP 6 — TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer(
        formatted_input,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    # --------------------------------------------------------
    # STEP 7 — GENERATE
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    # --------------------------------------------------------
    # STEP 8 — DECODE
    # --------------------------------------------------------

    generated_ids = (
        outputs[0][inputs.input_ids.shape[1]:]
    )

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    # --------------------------------------------------------
    # STEP 9 — EXTRACT LABEL
    # --------------------------------------------------------

    prediction = extract_label(
        generated_text
    )

    return prediction, results


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛡️ PhishDetectAI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Email Detection",
        "📊 Model Evaluation"
    ]
)


# ============================================================
# PAGE 1 — EMAIL DETECTION
# ============================================================

if page == "🔍 Email Detection":

    st.title("🛡️ PhishDetectAI")

    st.write(
        "AI-powered phishing email detection using "
        "Fine-Tuned Qwen and Retrieval-Augmented Generation."
    )

    st.divider()

    st.subheader("📧 Analyze an Email")

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if "email_widget_key" not in st.session_state:
        st.session_state.email_widget_key = 0

    if "email_input" not in st.session_state:
        st.session_state.email_input = ""

    if "analysis_results_displayed" not in st.session_state:
        st.session_state.analysis_results_displayed = False

    if "qwen_prediction" not in st.session_state:
        st.session_state.qwen_prediction = None

    if "rag_prediction" not in st.session_state:
        st.session_state.rag_prediction = None

    if "rag_results" not in st.session_state:
        st.session_state.rag_results = None

    # --------------------------------------------------------
    # DYNAMIC TEXT AREA KEY
    #
    # Changing the key forces Streamlit to create a new
    # empty text area after Clear Email.
    # --------------------------------------------------------

    email_widget_key = (
        f"email_text_{st.session_state.email_widget_key}"
    )

    email_text = st.text_area(
        "Enter the email content:",
        height=250,
        placeholder=(
            "Subject: Your account requires verification\n\n"
            "Body: Please click the link below..."
        ),
        key=email_widget_key
    )

    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    col_analyze, col_clear = st.columns([0.7, 0.3])

    with col_analyze:

        analyze_button = st.button(
            "🔍 Analyze Email",
            type="primary",
            use_container_width=True
        )

    with col_clear:

        clear_button = st.button(
            "Clear Email",
            use_container_width=True
        )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    if analyze_button:

        if not email_text.strip():

            st.warning(
                "Please enter an email before analyzing."
            )

            st.session_state.analysis_results_displayed = False

        else:

            st.session_state.email_input = email_text

            with st.spinner(
                "Analyzing email with both pipelines..."
            ):

                # --------------------------------------------
                # FINE-TUNED QWEN
                # --------------------------------------------

                qwen_prediction = predict_with_qwen(
                    email_text
                )

                # --------------------------------------------
                # RAG + QWEN
                # --------------------------------------------

                rag_prediction, rag_results = (
                    predict_with_rag_qwen(
                        email_text,
                        n_results=3
                    )
                )

            # ----------------------------------------------
            # SAVE RESULTS
            # ----------------------------------------------

            st.session_state.qwen_prediction = (
                qwen_prediction
            )

            st.session_state.rag_prediction = (
                rag_prediction
            )

            st.session_state.rag_results = (
                rag_results
            )

            st.session_state.analysis_results_displayed = True

    # --------------------------------------------------------
    # CLEAR EMAIL
    # --------------------------------------------------------

    if clear_button:

        # Clear stored email
        st.session_state.email_input = ""

        # Clear predictions
        st.session_state.qwen_prediction = None

        st.session_state.rag_prediction = None

        st.session_state.rag_results = None

        # Hide results
        st.session_state.analysis_results_displayed = False

        # Change widget key
        st.session_state.email_widget_key += 1

        # Rerun
        st.rerun()

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    if st.session_state.analysis_results_displayed:

        st.divider()

        st.subheader("🤖 Model Results")

        col1_res, col2_res = st.columns(2)

        # ====================================================
        # FINE-TUNED QWEN
        # ====================================================

        with col1_res:

            st.markdown(
                "### 🔵 Fine-Tuned Qwen"
            )

            if (
                st.session_state.qwen_prediction
                == "phishing"
            ):

                st.error(
                    "🚨 PHISHING"
                )

            elif (
                st.session_state.qwen_prediction
                == "legitimate"
            ):

                st.success(
                    "✅ LEGITIMATE"
                )

            else:

                st.warning(
                    "⚠️ INVALID RESPONSE"
                )

        # ====================================================
        # RAG + QWEN
        # ====================================================

        with col2_res:

            st.markdown(
                "### 🟣 RAG + Fine-Tuned Qwen"
            )

            if (
                st.session_state.rag_prediction
                == "phishing"
            ):

                st.error(
                    "🚨 PHISHING"
                )

            elif (
                st.session_state.rag_prediction
                == "legitimate"
            ):

                st.success(
                    "✅ LEGITIMATE"
                )

            else:

                st.warning(
                    "⚠️ INVALID RESPONSE"
                )

        # ====================================================
        # RAG EVIDENCE
        # ====================================================

        st.divider()

        st.subheader(
            "🔎 RAG Retrieved Evidence"
        )

        st.caption(
            "These emails were retrieved from the ChromaDB "
            "knowledge base and provided to Qwen as "
            "supporting evidence."
        )

        rag_results = st.session_state.rag_results

        if rag_results is not None:

            retrieved_count = len(
                rag_results["documents"][0]
            )

            for i in range(retrieved_count):

                label = (
                    rag_results["metadatas"][0][i]["label"]
                )

                distance = (
                    rag_results["distances"][0][i]
                )

                document = (
                    rag_results["documents"][0][i]
                )

                with st.expander(
                    f"Evidence {i + 1} — {label}"
                ):

                    st.write(
                        f"**Retrieved label:** {label}"
                    )

                    st.write(
                        f"**Distance:** {distance:.4f}"
                    )

                    st.text_area(
                        f"Retrieved email {i + 1}",
                        document,
                        height=180,
                        key=f"evidence_{i}"
                    )


# ============================================================
# PAGE 2 — MODEL EVALUATION
# ============================================================

elif page == "📊 Model Evaluation":

    st.title("📊 Model Evaluation")

    st.write(
        "Performance comparison of the three phishing "
        "detection approaches."
    )

    st.divider()

    st.subheader(
        "📈 Model Comparison on 2,977 Emails"
    )

    st.write(
        "All three models were evaluated on the same "
        "2,977-email test set."
    )

    # --------------------------------------------------------
    # SAVED EVALUATION RESULTS
    # --------------------------------------------------------

    comparison_data = {
        "Model": [
            "TF-IDF + Logistic Regression",
            "Fine-Tuned Qwen",
            "RAG + Qwen"
        ],

        "Accuracy": [
            "98.62%",
            "99.20%",
            "98.02%"
        ],

        "Precision": [
            "97.50%",
            "99.65%",
            "97.15%"
        ],

        "Recall": [
            "99.65%",
            "98.82%",
            "98.70%"
        ],

        "F1-Score": [
            "98.56%",
            "99.23%",
            "97.92%"
        ],

        "Invalid Predictions": [
            0,
            242,
            55
        ]
    }

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # METRIC EXPLANATION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📚 Evaluation Metrics"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **Accuracy**

            Percentage of predictions that were correct.

            **Precision**

            Of the emails predicted as phishing,
            how many were actually phishing.

            **Recall**

            Of the actual phishing emails,
            how many were detected.
            """
        )

    with col2:

        st.markdown(
            """
            **F1-Score**

            A combined measure of precision and recall.

            **Invalid Predictions**

            Model responses that did not produce
            a valid `phishing` or `legitimate` label.
            """
        )

    # --------------------------------------------------------
    # EVALUATION SETUP
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🧪 Evaluation Setup"
    )

    st.write(
        """
        **Test set:** 2,977 emails

        **Models compared:**
        - TF-IDF + Logistic Regression
        - Fine-Tuned Qwen
        - RAG + Fine-Tuned Qwen

        The same test split was used for the comparison
        to make the model results directly comparable.
        """
    )

    st.caption(
        "Results shown here were calculated previously in "
        "evaluation.ipynb. Opening this page does not "
        "rerun model evaluation."
    )
