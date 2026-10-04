"""
StudyBuddy: An interactive study assistant built for a friend.
Hacktoberfest Weekend Challenge: Build for a Friend.

Uses Groq / Grok API + Local HuggingFace sentence-transformer embeddings + FAISS vector store.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Search and load .env from current directory or parent directory
load_dotenv()
if not os.getenv("GROQ_API_KEY") and not os.getenv("GROK_API_KEY"):
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from pdf_processor import extract_text_from_pdfs, chunk_documents
from rag import (
    get_embedding_function,
    create_vector_store,
    get_llm,
    detect_provider,
    ask_question,
    explain_like_15,
    generate_quiz,
    generate_quick_revision,
    PROVIDER_CONFIG,
)

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StudyBuddy — Built for a Friend",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, beginner-friendly aesthetics
st.markdown(
    """
    <style>
    /* Clean typography and container spacing */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Header badge styling */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.85rem;
    }
    
    /* Friend story highlight card */
    .story-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 4px solid #3b82f6;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        color: #334155;
    }

    /* Source tag styling */
    .source-tag {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.80rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_docs_count" not in st.session_state:
    st.session_state.processed_docs_count = 0
if "processed_chunks_count" not in st.session_state:
    st.session_state.processed_chunks_count = 0
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_user_answers" not in st.session_state:
    st.session_state.quiz_user_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "revision_sheet" not in st.session_state:
    st.session_state.revision_sheet = None


# -----------------------------------------------------------------------------
# Sidebar: Setup, Model, File Upload, and Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 StudyBuddy")
    st.caption("Hacktoberfest: *Build for a Friend*")
    st.markdown("---")

    # 1. Detect Existing Environment Key
    env_groq = os.getenv("GROQ_API_KEY", "")
    env_grok = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or ""
    initial_key = env_groq or env_grok

    initial_provider = "groq" if (env_groq or not env_grok) else "xai"
    if initial_key.startswith("xai-"):
        initial_provider = "xai"
    elif initial_key.startswith("gsk_"):
        initial_provider = "groq"

    # Provider Selection
    provider_choice = st.selectbox(
        "AI Brain Provider",
        options=["Groq (Open-Weight Fast Inference)", "xAI Grok"],
        index=0 if initial_provider == "groq" else 1,
        help="Groq runs open-weight models at 800+ tokens/sec. xAI provides Grok models.",
    )
    current_provider = "groq" if "Groq" in provider_choice else "xai"

    # API Key Input
    api_key_input = st.text_input(
        f"{'Groq' if current_provider == 'groq' else 'Grok'} API Key",
        value=env_groq if current_provider == "groq" else env_grok,
        type="password",
        help="Reads automatically from .env or paste your key here.",
        placeholder="gsk_..." if current_provider == "groq" else "xai-...",
    )

    # Model Selection for selected provider
    available_models = PROVIDER_CONFIG[current_provider]["models"]
    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        index=0,
        help="Open-weight model powering answer generation, quizzes, and revision.",
    )

    st.markdown("---")

    # File Upload
    st.subheader("📁 Upload Notes")
    uploaded_files = st.file_uploader(
        "Choose PDF notes or slides",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple lecture notes, textbook chapters, or revision sheets.",
    )

    # Action Buttons
    process_btn = st.button("⚡ Process Notes", type="primary", use_container_width=True)

    if st.button("🗑️ Clear Chat / Reset", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.quiz_data = None
        st.session_state.quiz_user_answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.revision_sheet = None
        st.rerun()

    st.markdown("---")

    # Status & Index Stats
    if st.session_state.vector_store is not None:
        st.success(
            f"✅ **Notes Indexed!**\n\n"
            f"- Pages read: **{st.session_state.processed_docs_count}**\n"
            f"- Semantic chunks: **{st.session_state.processed_chunks_count}**"
        )
    else:
        st.info("ℹ️ Upload PDF(s) and click **Process Notes** to start.")

    st.markdown("---")
    st.caption(
        "🔒 **Local Embeddings**: Chunks are embedded locally using HuggingFace `all-MiniLM-L6-v2`. "
        "Only relevant chunks for your specific queries are sent to the AI model."
    )


# -----------------------------------------------------------------------------
# Notes Processing Handler
# -----------------------------------------------------------------------------
if process_btn:
    if not uploaded_files:
        st.sidebar.error("⚠️ Please select at least one PDF file first.")
    else:
        with st.status("Processing your study notes...", expanded=True) as status:
            try:
                # Step 1: Extract Text
                st.write("📄 Reading pages from uploaded PDF(s)...")
                raw_docs, warnings = extract_text_from_pdfs(uploaded_files)

                for w in warnings:
                    st.warning(w)

                if not raw_docs:
                    status.update(label="❌ No readable text found in PDFs.", state="error")
                    st.error(
                        "Could not extract any readable text from the uploaded files. "
                        "Please check if the PDFs are scanned images or protected."
                    )
                else:
                    # Step 2: Chunk Documents
                    st.write(f"✂️ Extracted {len(raw_docs)} pages. Splitting into semantic chunks...")
                    chunks = chunk_documents(raw_docs, chunk_size=900, chunk_overlap=120)

                    # Step 3: Local HuggingFace Embeddings
                    st.write("🧠 Loading local HuggingFace embeddings (`all-MiniLM-L6-v2`)...")
                    if st.session_state.embedding_model is None:
                        st.session_state.embedding_model = get_embedding_function()

                    # Step 4: Build FAISS Vector Index
                    st.write(f"📦 Generating FAISS vector index for {len(chunks)} chunks...")
                    vector_store = create_vector_store(chunks, st.session_state.embedding_model)

                    # Save to session
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_docs_count = len(raw_docs)
                    st.session_state.processed_chunks_count = len(chunks)

                    status.update(
                        label=f"✅ Ready! Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s).",
                        state="complete",
                    )
                    st.toast("Notes processed successfully! Click the tabs below to study.")

            except Exception as e:
                status.update(label="❌ Processing failed.", state="error")
                st.error(f"An error occurred while processing notes: {str(e)}")


# -----------------------------------------------------------------------------
# Main Application Header & Story Banner
# -----------------------------------------------------------------------------
st.title("📚 StudyBuddy")
st.markdown("*Your personal study assistant, built for a friend.*")

# Privacy status indicator
st.markdown(
    '<div class="status-badge">🟢 Local HuggingFace Embeddings + Open-Weight Brain — Fast, Private RAG</div>',
    unsafe_allow_html=True,
)

# Challenge Story Card
st.markdown(
    """
    <div class="story-card">
        <strong>💡 Built for a Friend:</strong>
        <em>"My friend spends a lot of time going through lengthy lecture PDFs before exams. 
        I wanted to build something small that could turn those notes into an interactive study assistant 
        instead of making them search through hundreds of pages."</em>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4 Main Feature Tabs
tab_ask, tab_explain, tab_quiz, tab_revision = st.tabs(
    ["💬 Ask", "🧠 Explain", "📝 Quiz", "⚡ Quick Revision"]
)


# Helper function to obtain LLM instance with user-friendly errors
def get_llm_or_warn():
    if not api_key_input:
        st.warning(f"⚠️ Please enter your {current_provider.upper()} API key in the sidebar.")
        return None
    try:
        return get_llm(
            api_key=api_key_input,
            provider=current_provider,
            model_name=selected_model,
        )
    except Exception as err:
        st.error(f"Error initializing AI model: {str(err)}")
        return None


# -----------------------------------------------------------------------------
# TAB 1: 💬 Ask Questions
# -----------------------------------------------------------------------------
with tab_ask:
    st.subheader("💬 Ask Your Notes")
    st.caption("Ask questions about concepts, definitions, formulas, or slides. Answers are grounded in your materials.")

    if st.session_state.vector_store is None:
        st.info("👈 Upload your PDF notes in the sidebar and click **Process Notes** to start asking questions.")
    else:
        # Quick suggestion chips
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        sample_q1 = cols[0].button("🔍 What are the core topics?", key="btn_q1", use_container_width=True)
        sample_q2 = cols[1].button("⚖️ What are the main trade-offs?", key="btn_q2", use_container_width=True)
        sample_q3 = cols[2].button("📝 Summarize key definitions", key="btn_q3", use_container_width=True)

        user_input_prompt = None
        if sample_q1:
            user_input_prompt = "What are the core topics covered in these notes?"
        elif sample_q2:
            user_input_prompt = "What are the main trade-offs, advantages, and disadvantages discussed in the notes?"
        elif sample_q3:
            user_input_prompt = "Summarize the key definitions and formulas in these notes."

        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📄 Retrieved Sources & Document Pages"):
                        for s in msg["sources"]:
                            st.markdown(f"- `{s}`")
                        if "chunks" in msg and msg["chunks"]:
                            st.markdown("**Context Chunks:**")
                            for idx, c in enumerate(msg["chunks"], 1):
                                st.markdown(f"**Chunk #{idx}** ({c['source']} — Page {c['page']}):")
                                st.caption(c["content"])

        # Chat input box
        chat_query = st.chat_input("Ask a question about your study material (e.g., 'What is overfitting?')...")
        effective_query = user_input_prompt or chat_query

        if effective_query:
            if not effective_query.strip():
                st.warning("Please type a valid question.")
            else:
                llm = get_llm_or_warn()
                if llm:
                    st.session_state.chat_history.append({"role": "user", "content": effective_query})
                    with st.chat_message("user"):
                        st.markdown(effective_query)

                    with st.chat_message("assistant"):
                        with st.spinner("Searching notes and generating answer..."):
                            try:
                                result = ask_question(
                                    vector_store=st.session_state.vector_store,
                                    llm=llm,
                                    question=effective_query,
                                    top_k=4,
                                )
                                answer_text = result["answer"]
                                sources = result["sources"]
                                chunks = result["chunks"]

                                st.markdown(answer_text)

                                if sources:
                                    with st.expander("📄 Retrieved Sources & Document Pages"):
                                        for s in sources:
                                            st.markdown(f"- `{s}`")
                                        st.markdown("**Context Chunks:**")
                                        for idx, c in enumerate(chunks, 1):
                                            st.markdown(f"**Chunk #{idx}** ({c['source']} — Page {c['page']}):")
                                            st.caption(c["content"])

                                # Save assistant message
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": answer_text,
                                    "sources": sources,
                                    "chunks": chunks,
                                })

                            except Exception as e:
                                st.error(f"Failed to generate answer: {str(e)}")


# -----------------------------------------------------------------------------
# TAB 2: 🧠 Explain Simply ("Explain Like I'm 15")
# -----------------------------------------------------------------------------
with tab_explain:
    st.subheader("🧠 Explain Like I'm 15")
    st.caption("Struggling with a dense or confusing topic? Let the AI break it down with relatable real-world analogies.")

    if st.session_state.vector_store is None:
        st.info("👈 Upload your PDF notes in the sidebar and click **Process Notes** first.")
    else:
        concept_input = st.text_input(
            "Concept or topic to simplify:",
            placeholder="e.g., Overfitting, Backpropagation, CAP Theorem, Pointers, Normal Distribution",
        )

        explain_btn = st.button("🚀 Explain Simply (ELI15)", type="primary")

        if explain_btn:
            if not concept_input or not concept_input.strip():
                st.warning("Please enter a concept name to explain.")
            else:
                llm = get_llm_or_warn()
                if llm:
                    with st.spinner(f"Simplifying '{concept_input}' for high school level..."):
                        try:
                            result = explain_like_15(
                                vector_store=st.session_state.vector_store,
                                llm=llm,
                                concept=concept_input.strip(),
                                top_k=4,
                            )
                            st.markdown("### 💡 Simplified Breakdown")
                            st.markdown(result["explanation"])

                            if result["sources"]:
                                with st.expander("📄 Notes Referenced"):
                                    for s in result["sources"]:
                                        st.markdown(f"- `{s}`")
                        except Exception as e:
                            st.error(f"Error generating explanation: {str(e)}")


# -----------------------------------------------------------------------------
# TAB 3: 📝 Generate Quiz
# -----------------------------------------------------------------------------
with tab_quiz:
    st.subheader("📝 Practice Quiz")
    st.caption("Test your recall with 5 multiple-choice questions synthesized straight from your lecture notes.")

    if st.session_state.vector_store is None:
        st.info("👈 Upload your PDF notes in the sidebar and click **Process Notes** first.")
    else:
        col_topic, col_btn = st.columns([3, 1])
        quiz_topic = col_topic.text_input(
            "Quiz Focus / Subtopic (Optional):",
            placeholder="Leave blank for entire material, or specify e.g., 'Chapter 3' or 'Algorithms'",
        )
        gen_quiz_btn = col_btn.button("🎲 Generate Quiz", type="primary", use_container_width=True)

        if gen_quiz_btn:
            llm = get_llm_or_warn()
            if llm:
                with st.spinner("Generating 5 high-yield multiple-choice questions..."):
                    try:
                        res = generate_quiz(
                            vector_store=st.session_state.vector_store,
                            llm=llm,
                            topic=quiz_topic.strip() if quiz_topic else "Key concepts from the notes",
                        )
                        st.session_state.quiz_data = res.get("quiz_data") or []
                        st.session_state.quiz_raw = res.get("raw_text", "")
                        st.session_state.quiz_sources = res.get("context_sources", [])
                        st.session_state.quiz_user_answers = {}
                        st.session_state.quiz_submitted = False
                    except Exception as e:
                        st.error(f"Failed to generate quiz: {str(e)}")

        # Display Quiz if available
        if st.session_state.quiz_data and len(st.session_state.quiz_data) > 0:
            st.markdown("---")
            st.markdown("#### 🎯 Answer the 5 questions below:")

            with st.form("quiz_form"):
                for q_idx, item in enumerate(st.session_state.quiz_data, start=1):
                    st.markdown(f"**Question {q_idx}: {item.get('question', '')}**")
                    options = item.get("options", [])

                    selected = st.radio(
                        label=f"Select your answer for Question {q_idx}",
                        options=options,
                        key=f"q_{q_idx}",
                        index=None,
                        label_visibility="collapsed",
                    )
                    st.session_state.quiz_user_answers[q_idx] = selected
                    st.markdown("<br>", unsafe_allow_html=True)

                submit_answers = st.form_submit_button("✅ Submit Quiz & See Results", type="primary")

            if submit_answers:
                st.session_state.quiz_submitted = True

            if st.session_state.quiz_submitted:
                st.markdown("---")
                st.markdown("### 📊 Quiz Results & Review")

                correct_count = 0
                total_q = len(st.session_state.quiz_data)

                for q_idx, item in enumerate(st.session_state.quiz_data, start=1):
                    user_choice = st.session_state.quiz_user_answers.get(q_idx)
                    correct_letter = item.get("correct_answer", "").strip().upper()
                    explanation = item.get("explanation", "")

                    is_correct = False
                    if user_choice and user_choice.strip().startswith(correct_letter):
                        is_correct = True
                        correct_count += 1

                    status_emoji = "✅ Correct!" if is_correct else "❌ Needs Review"

                    with st.expander(f"Question {q_idx}: {status_emoji}", expanded=True):
                        st.markdown(f"**{item.get('question', '')}**")
                        st.markdown(f"- **Your Answer:** {user_choice if user_choice else '*(None selected)*'}")
                        st.markdown(f"- **Correct Answer:** **Option {correct_letter}**")
                        st.info(f"💡 **Explanation:** {explanation}")

                score_pct = int((correct_count / total_q) * 100) if total_q > 0 else 0
                if score_pct >= 80:
                    st.balloons()
                    st.success(f"🎉 **Great job! You scored {correct_count}/{total_q} ({score_pct}%)!** You're exam-ready on this topic.")
                elif score_pct >= 60:
                    st.warning(f"👍 **Good effort! You scored {correct_count}/{total_q} ({score_pct}%).** Review the explanations above for the questions you missed.")
                else:
                    st.error(f"📖 **You scored {correct_count}/{total_q} ({score_pct}%).** Check the 'Explain' or 'Quick Revision' tabs to reinforce these topics before trying again!")

        elif hasattr(st.session_state, "quiz_raw") and st.session_state.quiz_raw:
            st.markdown("### 📝 Generated Quiz Questions")
            st.markdown(st.session_state.quiz_raw)


# -----------------------------------------------------------------------------
# TAB 4: ⚡ Quick Revision
# -----------------------------------------------------------------------------
with tab_revision:
    st.subheader("⚡ Quick Revision")
    st.caption("Generate a high-impact cheat sheet covering core concepts, formulas, definitions, and exam traps.")

    if st.session_state.vector_store is None:
        st.info("👈 Upload your PDF notes in the sidebar and click **Process Notes** first.")
    else:
        if st.button("⚡ Generate Quick Revision Sheet", type="primary"):
            llm = get_llm_or_warn()
            if llm:
                with st.spinner("Extracting high-yield exam takeaways..."):
                    try:
                        res = generate_quick_revision(
                            vector_store=st.session_state.vector_store,
                            llm=llm,
                            sample_size=8,
                        )
                        st.session_state.revision_sheet = res.get("revision_content", "")
                        st.session_state.revision_sources = res.get("sources", [])
                    except Exception as e:
                        st.error(f"Failed to generate revision sheet: {str(e)}")

        if st.session_state.revision_sheet:
            st.markdown("---")
            st.markdown(st.session_state.revision_sheet)

            st.download_button(
                label="📥 Download Revision Sheet (Markdown)",
                data=st.session_state.revision_sheet,
                file_name="studybuddy_quick_revision.md",
                mime="text/markdown",
            )
