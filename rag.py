"""
RAG (Retrieval-Augmented Generation) pipeline for StudyBuddy.

Supports:

- Groq Cloud API
- xAI Grok API
- Local HuggingFace sentence-transformers
- Local FAISS vector store
- Question answering
- Explain Like I'm 15
- Quiz generation
- Quick revision
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()


# ============================================================
# EMBEDDINGS
# ============================================================

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


try:
    from langchain_huggingface import HuggingFaceEmbeddings

except ImportError:

    try:
        from langchain_community.embeddings import (
            HuggingFaceEmbeddings
        )

    except ImportError:

        HuggingFaceEmbeddings = None


# ============================================================
# FAISS
# ============================================================

try:
    from langchain_community.vectorstores import FAISS

except ImportError:

    FAISS = None


# ============================================================
# PROMPTS
# ============================================================

from prompts import (
    QA_PROMPT,
    ELI15_PROMPT,
    QUIZ_PROMPT,
    QUICK_REVISION_PROMPT,
)


# ============================================================
# PROVIDER CONFIGURATION
# ============================================================

PROVIDER_CONFIG = {

    "groq": {

        "name": "Groq Cloud",

        "models": [
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
        ],

        "default_model": "openai/gpt-oss-120b",

        "env_var": "GROQ_API_KEY",
    },

    "xai": {

        "name": "xAI Grok",

        "models": [
            "grok-3-mini",
            "grok-3",
        ],

        "default_model": "grok-3-mini",

        "env_var": "XAI_API_KEY",
    },
}


# ============================================================
# PROVIDER DETECTION
# ============================================================

def detect_provider(api_key: str) -> str:
    """
    Detect provider based on API key format.
    """

    if not api_key:

        groq_key = os.getenv(
            "GROQ_API_KEY",
            ""
        )

        xai_key = os.getenv(
            "XAI_API_KEY",
            ""
        )

        if groq_key:
            return "groq"

        if xai_key:
            return "xai"

        return "groq"


    key = api_key.strip()


    if key.startswith("gsk_"):
        return "groq"


    if key.startswith("xai-"):
        return "xai"


    # Default
    return "groq"


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def get_embedding_function(
    model_name: str = DEFAULT_EMBEDDING_MODEL
):
    """
    Create local HuggingFace embedding model.

    Embeddings run locally.
    """

    if HuggingFaceEmbeddings is None:

        raise ImportError(
            "HuggingFaceEmbeddings is not installed.\n\n"
            "Install using:\n"
            "pip install -U sentence-transformers "
            "langchain-huggingface"
        )


    try:

        embeddings = HuggingFaceEmbeddings(

            model_name=model_name,

            model_kwargs={
                "device": "cpu"
            },

            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        return embeddings

    except Exception as exc:

        raise RuntimeError(
            f"Failed to initialize embedding model: {exc}"
        ) from exc


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(
    chunks: List[Document],
    embedding_model=None
) -> Any:
    """
    Build an in-memory FAISS vector store.
    """

    if not chunks:

        raise ValueError(
            "Cannot create vector store because "
            "no document chunks were provided."
        )


    if FAISS is None:

        raise ImportError(
            "FAISS is not installed.\n\n"
            "Install using:\n"
            "pip install -U faiss-cpu "
            "langchain-community"
        )


    if embedding_model is None:

        embedding_model = (
            get_embedding_function()
        )


    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )


    return vector_store


# ============================================================
# GET LLM
# ============================================================

def get_llm(
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.3,
):
    """
    Create an LLM for Groq or xAI.

    Provider can be:
        - groq
        - xai
    """

    # --------------------------------------------------------
    # Resolve API key
    # --------------------------------------------------------

    resolved_key = api_key


    if not resolved_key:

        resolved_key = os.getenv(
            "GROQ_API_KEY"
        )


    if not resolved_key:

        resolved_key = os.getenv(
            "XAI_API_KEY"
        )


    if not resolved_key:

        raise ValueError(
            "API key is missing.\n\n"
            "Provide an API key in the sidebar "
            "or add GROQ_API_KEY / XAI_API_KEY "
            "to your .env file."
        )


    resolved_key = resolved_key.strip()


    # --------------------------------------------------------
    # Detect provider
    # --------------------------------------------------------

    if not provider:

        provider = detect_provider(
            resolved_key
        )


    provider = provider.lower()


    if provider not in PROVIDER_CONFIG:

        raise ValueError(
            f"Unsupported provider: {provider}\n"
            f"Available providers: "
            f"{list(PROVIDER_CONFIG.keys())}"
        )


    config = PROVIDER_CONFIG[
        provider
    ]


    selected_model = (
        model_name
        or config["default_model"]
    )


    # ========================================================
    # GROQ
    # ========================================================

    if provider == "groq":

        try:

            from langchain_groq import (
                ChatGroq
            )

        except ImportError as exc:

            raise ImportError(
                "langchain-groq is not installed.\n\n"
                "Install using:\n"
                "pip install -U langchain-groq"
            ) from exc


        return ChatGroq(

            model=selected_model,

            api_key=resolved_key,

            temperature=temperature,

            max_tokens=2048,
        )


    # ========================================================
    # XAI
    # ========================================================

    if provider == "xai":

        try:

            from langchain_openai import (
                ChatOpenAI
            )

        except ImportError as exc:

            raise ImportError(
                "langchain-openai is not installed.\n\n"
                "Install using:\n"
                "pip install -U langchain-openai"
            ) from exc


        return ChatOpenAI(

            model=selected_model,

            api_key=resolved_key,

            base_url="https://api.x.ai/v1",

            temperature=temperature,

            max_tokens=2048,
        )


    raise ValueError(
        f"Could not create LLM for provider: {provider}"
    )


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_context_docs(
    docs: List[Document]
) -> str:
    """
    Convert retrieved documents into a readable
    context string.
    """

    if not docs:

        return "No relevant documents found."


    formatted_chunks = []


    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown Document"
        )


        page = doc.metadata.get(
            "page",
            "?"
        )


        content = (
            doc.page_content
            or ""
        ).strip()


        formatted_chunks.append(

            f"--- "
            f"[Document: {source} | Page: {page}] "
            f"---\n"
            f"{content}"
        )


    return "\n\n".join(
        formatted_chunks
    )


# ============================================================
# EXTRACT RESPONSE TEXT
# ============================================================

def get_response_text(
    response: Any
) -> str:
    """
    Safely extract text from LangChain AIMessage.
    """

    content = getattr(
        response,
        "content",
        response
    )


    if isinstance(
        content,
        str
    ):

        return content


    if isinstance(
        content,
        list
    ):

        parts = []


        for item in content:

            if isinstance(
                item,
                dict
            ):

                text = item.get(
                    "text"
                )

                if text:
                    parts.append(
                        str(text)
                    )

            else:

                parts.append(
                    str(item)
                )


        return "\n".join(
            parts
        )


    return str(content)


# ============================================================
# BUILD SOURCE INFORMATION
# ============================================================

def build_sources(
    docs: List[Document]
):
    """
    Build source list and chunk information.
    """

    sources = []

    chunks = []


    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Document"
        )


        page = doc.metadata.get(
            "page",
            "?"
        )


        reference = (
            f"{source} (Page {page})"
        )


        if reference not in sources:

            sources.append(
                reference
            )


        chunks.append(

            {
                "source": source,

                "page": page,

                "content": (
                    doc.page_content
                    or ""
                ),
            }
        )


    return sources, chunks


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(
    vector_store: Any,
    llm: Any,
    question: str,
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Answer a question using retrieved notes.
    """

    if not question or not question.strip():

        return {

            "answer":
                "Please enter a question "
                "regarding your notes.",

            "sources": [],

            "chunks": [],
        }


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieved_docs = (
        vector_store.similarity_search(
            question.strip(),
            k=top_k
        )
    )


    if not retrieved_docs:

        return {

            "answer":
                "Based on your uploaded notes, "
                "no relevant context was found.",

            "sources": [],

            "chunks": [],
        }


    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context_str = format_context_docs(
        retrieved_docs
    )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt_value = QA_PROMPT.format(

        context=context_str,

        question=question.strip()
    )


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    response = llm.invoke(
        prompt_value
    )


    answer_text = get_response_text(
        response
    )


    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources, chunks = build_sources(
        retrieved_docs
    )


    return {

        "answer": answer_text,

        "sources": sources,

        "chunks": chunks,
    }


# ============================================================
# EXPLAIN LIKE I'M 15
# ============================================================

def explain_like_15(
    vector_store: Any,
    llm: Any,
    concept: str,
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Explain a concept in simple language
    using uploaded notes.
    """

    if not concept or not concept.strip():

        return {

            "explanation":
                "Please enter a concept "
                "or topic to explain.",

            "sources": [],

            "chunks": [],
        }


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieved_docs = (
        vector_store.similarity_search(
            concept.strip(),
            k=top_k
        )
    )


    context_str = (

        format_context_docs(
            retrieved_docs
        )

        if retrieved_docs

        else "No specific notes found."
    )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt_value = ELI15_PROMPT.format(

        context=context_str,

        concept=concept.strip()
    )


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    response = llm.invoke(
        prompt_value
    )


    explanation_text = (
        get_response_text(response)
    )


    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources, chunks = build_sources(
        retrieved_docs
    )


    return {

        "explanation":
            explanation_text,

        "sources":
            sources,

        "chunks":
            chunks,
    }


# ============================================================
# PARSE QUIZ JSON
# ============================================================

def parse_quiz_json(
    raw_text: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Extract a JSON array from an LLM response.

    Supports:
        [...]
        ```json
        [...]
        ```
    """

    if not raw_text:

        return None


    text = raw_text.strip()


    # --------------------------------------------------------
    # Remove markdown code fence
    # --------------------------------------------------------

    fenced_match = re.search(

        r"```(?:json)?\s*(.*?)\s*```",

        text,

        re.DOTALL | re.IGNORECASE
    )


    if fenced_match:

        candidate = (
            fenced_match.group(1)
            .strip()
        )

        try:

            parsed = json.loads(
                candidate
            )

            if isinstance(
                parsed,
                list
            ):

                return parsed

        except json.JSONDecodeError:

            pass


    # --------------------------------------------------------
    # Find JSON array
    # --------------------------------------------------------

    start = text.find("[")
    end = text.rfind("]")


    if start != -1 and end != -1:

        candidate = text[
            start:end + 1
        ]


        try:

            parsed = json.loads(
                candidate
            )


            if isinstance(
                parsed,
                list
            ):

                return parsed

        except json.JSONDecodeError:

            pass


    return None


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz(
    vector_store: Any,
    llm: Any,
    topic: str = "Key concepts from the notes",
    num_samples: int = 6
) -> Dict[str, Any]:
    """
    Generate multiple-choice questions
    from uploaded notes.
    """

    search_query = (

        topic.strip()

        if topic
        and topic.strip()

        else
        (
            "important concepts "
            "definitions algorithms "
            "formulas rules"
        )
    )


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieved_docs = (
        vector_store.similarity_search(
            search_query,
            k=num_samples
        )
    )


    if not retrieved_docs:

        return {

            "quiz_data": [],

            "raw_text":
                "Unable to find notes "
                "content to generate questions.",

            "error":
                "No notes retrieved.",
        }


    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context_str = format_context_docs(
        retrieved_docs
    )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt_value = QUIZ_PROMPT.format(

        context=context_str,

        topic=topic
    )


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    response = llm.invoke(
        prompt_value
    )


    raw_text = get_response_text(
        response
    )


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    quiz_data = parse_quiz_json(
        raw_text
    )


    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = list(
        dict.fromkeys(

            [
                d.metadata.get(
                    "source",
                    "Notes"
                )

                for d in retrieved_docs
            ]
        )
    )


    return {

        "quiz_data":
            quiz_data or [],

        "raw_text":
            raw_text,

        "context_sources":
            sources,
    }


# ============================================================
# QUICK REVISION
# ============================================================

def generate_quick_revision(
    vector_store: Any,
    llm: Any,
    sample_size: int = 8
) -> Dict[str, Any]:
    """
    Generate a concise exam revision sheet.
    """

    query = (
        "summary overview key concepts "
        "formulas mistakes definitions "
        "rules important topics"
    )


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieved_docs = (
        vector_store.similarity_search(
            query,
            k=sample_size
        )
    )


    if not retrieved_docs:

        return {

            "revision_content":
                "No content found "
                "in uploaded notes.",

            "sources": [],
        }


    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context_str = format_context_docs(
        retrieved_docs
    )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt_value = (
        QUICK_REVISION_PROMPT.format(
            context=context_str
        )
    )


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    response = llm.invoke(
        prompt_value
    )


    content = get_response_text(
        response
    )


    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = list(
        dict.fromkeys(

            [
                d.metadata.get(
                    "source",
                    "Notes"
                )

                for d in retrieved_docs
            ]
        )
    )


    return {

        "revision_content":
            content,

        "sources":
            sources,
    }