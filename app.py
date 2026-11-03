import os
import pandas as pd
import streamlit as st
import torch
import laya

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------------------------------------
# App configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Offline LLM Evaluation Framework",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Offline LLM Evaluation Framework")
st.caption("FLAN-T5 Small candidate model | Laya LLM-as-a-Judge")

DATA_DIR = "data"
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "knowledge_base.csv")
BENCHMARK_PATH = os.path.join(DATA_DIR, "final_evaluation_dataset.csv")
LAYA_RESULTS_PATH = os.path.join(DATA_DIR, "laya_evaluation_results.csv")

CANDIDATE_MODEL_NAME = "google/flan-t5-small"
JUDGE_MODEL_NAME = "convaiinnovations/laya"

DIMENSIONS = [
    "correctness",
    "relevance",
    "completeness",
    "conciseness",
    "faithfulness",
]

# Keep these rubrics aligned with the evaluation performed in Colab.
EVALUATION_QUESTIONS = {
    "correctness": {
        "type": "choice",
        "instructions": (
            "Evaluate factual accuracy only. Treat the supplied context as "
            "the authoritative source. Assess whether factual claims in the "
            "candidate answer are accurate. Do not penalize missing details; "
            "evaluate those under completeness."
        ),
        "criteria": {
            "pass": "All factual claims are accurate and consistent with the context.",
            "partial": "The main factual answer is correct but contains a minor factual imprecision.",
            "fail": "The main answer is factually incorrect or contradicts the context.",
        },
    },
    "relevance": {
        "type": "choice",
        "instructions": (
            "Evaluate whether the candidate answer addresses the user's actual "
            "question. Focus only on relevance, not completeness or factual accuracy."
        ),
        "criteria": {
            "pass": "The answer directly addresses the question.",
            "partial": "The answer addresses part of the question but also includes some unrelated information.",
            "fail": "The answer does not address the question or discusses an unrelated topic.",
        },
    },
    "completeness": {
        "type": "choice",
        "instructions": (
            "Evaluate whether the candidate answer contains all information "
            "required by the question. Check important conditions, restrictions, "
            "quantities and time periods."
        ),
        "criteria": {
            "pass": "All essential information required by the question is included.",
            "partial": "The main answer is present but one or more required details are missing.",
            "fail": "The answer omits the central information required to answer the question.",
        },
    },
    "conciseness": {
        "type": "choice",
        "instructions": (
            "Evaluate the efficiency of the candidate answer. Do not reward "
            "shortness alone. Assess whether the answer communicates its "
            "information without unnecessary repetition or irrelevant elaboration."
        ),
        "criteria": {
            "pass": "The answer is appropriately brief and communicates its information efficiently.",
            "partial": "The answer contains some unnecessary wording or repetition but remains reasonably clear.",
            "fail": "Excessive irrelevant or repetitive wording makes the answer inefficient or difficult to understand.",
        },
    },
    "faithfulness": {
        "type": "choice",
        "instructions": (
            "Evaluate whether every factual claim in the candidate answer is "
            "supported by the supplied context. Do not use outside knowledge. "
            "Distinguish supported claims from unsupported additions and contradictions."
        ),
        "criteria": {
            "pass": "All factual claims are supported by the supplied context.",
            "partial": "The main answer is supported but includes a minor unsupported detail.",
            "fail": "The answer contains materially unsupported claims or contradicts the supplied context.",
        },
    },
}


# ---------------------------------------------------------
# Cached model and data loading
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading FLAN-T5 Small...")
def load_candidate_model():
    tokenizer = AutoTokenizer.from_pretrained(CANDIDATE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(CANDIDATE_MODEL_NAME)
    model.to("cpu")
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading Laya judge...")
def load_judge_model():
    return laya.load(JUDGE_MODEL_NAME)


@st.cache_data
def load_knowledge_base():
    df = pd.read_csv(KNOWLEDGE_BASE_PATH)
    required = {"policy_id", "category", "policy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Knowledge base is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["policy"]).reset_index(drop=True)
    return df


@st.cache_resource
def build_retriever(policy_texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(policy_texts)
    return vectorizer, matrix


def retrieve_policy(question, kb_df):
    policy_texts = (
        kb_df["category"].fillna("").astype(str)
        + " "
        + kb_df["policy"].fillna("").astype(str)
    ).tolist()

    vectorizer, policy_matrix = build_retriever(tuple(policy_texts))
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, policy_matrix).flatten()

    best_index = int(scores.argmax())
    best_row = kb_df.iloc[best_index]

    return best_row, float(scores[best_index])


def generate_candidate_answer(question, context, tokenizer, model):
    prompt = f"""Answer the question using only the information provided in the context.

Context: {context}

Question: {question}

Answer:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def evaluate_with_laya(question, context, candidate_response, judge):
    state = {
        "question": question,
        "context": context,
        "golden_answer": "",
        "candidate_response": candidate_response,
    }

    result = judge.predict(
        state=state,
        questions=EVALUATION_QUESTIONS,
    )

    return result


# ---------------------------------------------------------
# Main interface
# ---------------------------------------------------------
live_tab, benchmark_tab, about_tab = st.tabs(
    ["Live Evaluation", "Benchmark Results", "About"]
)

with live_tab:
    st.subheader("Evaluate a customer-support question")
    st.write(
        "Enter a question. The app retrieves a policy, generates an answer "
        "with FLAN-T5 Small, and evaluates it with Laya."
    )

    question = st.text_area(
        "Your question",
        placeholder="Example: Can I change my delivery address after shipping?",
        height=90,
    )

    run_button = st.button(
        "Generate and evaluate",
        type="primary",
        disabled=not question.strip(),
    )

    if run_button:
        try:
            kb_df = load_knowledge_base()
            policy_row, retrieval_score = retrieve_policy(question.strip(), kb_df)
            context = str(policy_row["policy"])

            with st.spinner("Loading models and generating the candidate answer..."):
                tokenizer, candidate_model = load_candidate_model()
                candidate_response = generate_candidate_answer(
                    question.strip(),
                    context,
                    tokenizer,
                    candidate_model,
                )

            st.markdown("### Candidate model response")
            st.info(candidate_response if candidate_response else "The candidate model returned an empty response.")

            with st.expander("Retrieved policy context"):
                st.write(f"**Policy ID:** {policy_row['policy_id']}")
                st.write(f"**Category:** {policy_row['category']}")
                st.write(f"**Retrieval similarity:** {retrieval_score:.3f}")
                st.write(context)

            with st.spinner("Laya is evaluating the candidate response across five dimensions..."):
                judge = load_judge_model()
                judge_result = evaluate_with_laya(
                    question.strip(),
                    context,
                    candidate_response,
                    judge,
                )

            st.markdown("### Laya evaluation")
            answers = judge_result.get("answers", {})

            columns = st.columns(5)
            for column, dimension in zip(columns, DIMENSIONS):
                answer = answers.get(dimension, {})
                decision = answer.get("choice", "not available")
                probabilities = answer.get("probabilities", {})

                with column:
                    st.metric(dimension.capitalize(), decision.upper())
                    st.caption("Model-reported option probabilities")
                    for label in ["pass", "partial", "fail"]:
                        value = probabilities.get(label)
                        if value is not None:
                            st.write(f"{label.capitalize()}: {value:.1%}")
                            st.progress(float(value))

            st.warning(
                "Laya's checkpoint reported that some confidence values may be "
                "uncalibrated. Treat the displayed probabilities as model outputs, "
                "not verified probabilities of correctness."
            )

            usage = judge_result.get("usage", {})
            st.caption(
                f"Judge input tokens: {usage.get('input_tokens', 'N/A')} | "
                f"Input truncated: {usage.get('truncated', 'N/A')}"
            )

        except Exception as exc:
            st.error("The evaluation could not be completed.")
            st.exception(exc)


with benchmark_tab:
    st.subheader("Offline benchmark results")

    if os.path.exists(BENCHMARK_PATH):
        benchmark_df = pd.read_csv(BENCHMARK_PATH)

        st.metric("Benchmark test cases", len(benchmark_df))

        summary_rows = []
        for dimension in DIMENSIONS:
            column_name = f"{dimension}_decision"
            if column_name in benchmark_df.columns:
                counts = benchmark_df[column_name].value_counts()
                summary_rows.append({
                    "Dimension": dimension.capitalize(),
                    "Pass": int(counts.get("pass", 0)),
                    "Partial": int(counts.get("partial", 0)),
                    "Fail": int(counts.get("fail", 0)),
                })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).set_index("Dimension")
            st.markdown("### Decision distribution")
            st.dataframe(summary_df, use_container_width=True)
            st.bar_chart(summary_df)

        st.markdown("### Benchmark records")
        st.dataframe(benchmark_df, use_container_width=True)

        csv_bytes = benchmark_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download benchmark results",
            data=csv_bytes,
            file_name="final_evaluation_dataset.csv",
            mime="text/csv",
        )
    else:
        st.warning("Benchmark file not found. Add data/final_evaluation_dataset.csv to the repository.")

    if os.path.exists(LAYA_RESULTS_PATH):
        with st.expander("View raw Laya evaluation results"):
            laya_df = pd.read_csv(LAYA_RESULTS_PATH)
            st.dataframe(laya_df, use_container_width=True)


with about_tab:
    st.subheader("Project overview")
    st.markdown(
        """
        **Candidate model:** Google FLAN-T5 Small (`google/flan-t5-small`)

        **Judge model:** Laya (`convaiinnovations/laya`)

        **Evaluation dimensions:**
        - Correctness
        - Relevance
        - Completeness
        - Conciseness
        - Faithfulness

        The benchmark contains 100 test cases and 500 dimension-level judgments.
        The benchmark labels and Laya decisions should be treated as evaluation
        artifacts, not as independently verified human ground truth.
        """
    )
    st.caption(
        "Models are downloaded by the hosting server at startup and cached for "
        "the lifetime of the app process. No model weights need to be committed "
        "to this GitHub repository."
    )
