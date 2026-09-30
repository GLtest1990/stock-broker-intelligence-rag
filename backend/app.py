# Import necessary libraries
import logging
import os
import traceback
from flask import Flask, request, jsonify  # For creating the Flask API
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Import RAG utility functions
from rag_utils import retrieve, rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("broker-api")

# ---------------------------------------------------------------------------
# Configuration (can be overridden with environment variables at `docker run`)
# ---------------------------------------------------------------------------
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "vectordb")        # FAISS index packaged in the image
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"        # must match the model used to build the index
EMBED_DIMENSIONS = 384                                        # must match the dimensions of the index
DEFAULT_LLM = os.getenv("LLM_MODEL", "gpt-4o-mini")           # OpenAI answer-generation model
DEFAULT_K = 4                                                 # chunks retrieved per question
MAX_K = 10

# Initialize the Flask application
rag_api = Flask("Stock Broker Intelligence API")

# Resources loaded once at start-up. If something is wrong (missing key, missing index files)
# the API still starts and reports the problem through /health and the query endpoints,
# instead of the container exiting and the frontend only seeing a connection error.
embeddings = None
vectorstore = None
STARTUP_ERRORS = []

# 1) Embedding model (downloaded into the image at build time) and the persisted FAISS vector database
try:
    missing = [n for n in ("index.faiss", "index.pkl") if not os.path.isfile(os.path.join(VECTOR_DB_DIR, n))]
    if missing:
        raise RuntimeError(f"Vector database file(s) not found in '{VECTOR_DB_DIR}': {missing}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},   # same setting as used when the index was built
    )
    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # the index was created by us, so it is trusted
    )
    if vectorstore.index.d != EMBED_DIMENSIONS:
        raise RuntimeError(
            f"Index has {vectorstore.index.d} dimensions but the embedding model produces {EMBED_DIMENSIONS}. "
            "Rebuild the vector database with the same embedding model."
        )
    embeddings.embed_query("warm-up")   # so the first real question is not slow
    log.info("Vector DB loaded: %s vectors, %s dimensions", vectorstore.index.ntotal, vectorstore.index.d)
except Exception as e:
    vectorstore = None
    STARTUP_ERRORS.append(f"Vector database / embeddings: {type(e).__name__}: {e}")
    log.error("Vector database failed to load:\n%s", traceback.format_exc())

# 2) Credentials for answer generation. Embeddings are computed locally, so OpenAI is only used to write answers.
if not os.getenv("OPENAI_API_KEY"):
    STARTUP_ERRORS.append(
        "OPENAI_API_KEY is not set inside the container. "
        "Start the container with: -e OPENAI_API_KEY -e OPENAI_API_BASE (after exporting them in the terminal)."
    )

if STARTUP_ERRORS:
    log.error("API started in DEGRADED mode: %s", STARTUP_ERRORS)
else:
    log.info("API is ready.")


def not_ready():
    """Return an explanatory 503 when the vector database or the credentials are not available."""
    return jsonify({"error": "Backend is not ready.", "details": STARTUP_ERRORS}), 503


def read_params():
    """Read and validate the JSON body shared by the query endpoints."""
    data = request.get_json(silent=True) or {}

    query = str(data.get("query", "")).strip()
    if not query:
        raise ValueError("Field 'query' is required.")

    try:
        return {
            "query": query,
            "k": min(max(int(data.get("k", DEFAULT_K)), 1), MAX_K),
            "model_name": str(data.get("model_name", DEFAULT_LLM)),
            "temperature": float(data.get("temperature", 0.0)),
            "top_p": float(data.get("top_p", 1.0)),
            "max_tokens": int(data.get("max_tokens", 512)),
        }
    except (TypeError, ValueError):
        raise ValueError(
            "'k' and 'max_tokens' must be integers; 'temperature' and 'top_p' must be numbers."
        )


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """
    Handles GET requests to the root URL.
    """
    return "Welcome to the Stock Broker Intelligence RAG API!"


# ---------------------------------------------------------
# Health route
# ---------------------------------------------------------

@rag_api.get("/health")
def health():
    """
    Reports that the vector database is loaded. Add ?deep=1 to also test the local embedding
    model and the OpenAI chat call (the quickest way to diagnose a 500 error).
    """
    if vectorstore is None or STARTUP_ERRORS:
        return jsonify({"status": "degraded", "details": STARTUP_ERRORS}), 503

    info = {
        "status": "ok",
        "vectors": int(vectorstore.index.ntotal),
        "dimensions": int(vectorstore.index.d),
        "embedding_model": EMBED_MODEL,
        "llm_model": DEFAULT_LLM,
        "custom_openai_base": bool(os.getenv("OPENAI_API_BASE")),
    }
    if request.args.get("deep"):
        try:
            info["embedding_dimensions_returned"] = len(embeddings.embed_query("ping"))
            info["llm_reply"] = str(ChatOpenAI(model=DEFAULT_LLM).invoke("Reply with the single word: ok").content)[:50]
        except Exception as e:
            log.error("Deep health check failed:\n%s", traceback.format_exc())
            info["status"] = "error"
            info["error"] = f"{type(e).__name__}: {e}"
            return jsonify(info), 500
    return jsonify(info)


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """
    Handles POST requests to /v1/relevant_chunks.

    Example request:
    {
        "query": "What risks are mentioned in the SEC filings for Goldman Sachs?",
        "k": 3
    }

    Returns the top-k relevant chunks retrieved from the vector store.
    """
    if vectorstore is None:
        return not_ready()

    try:
        params = read_params()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    chunks = retrieve(query=params["query"], k=params["k"], vectorstore=vectorstore)

    return jsonify({
        "query": params["query"],
        "k": params["k"],
        "relevant_chunks": chunks
    })


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """
    Handles POST requests to /v1/answer_with_relevant_chunks.

    Example request:
    {
        "query": "What risks are mentioned in the SEC filings for Goldman Sachs?",
        "k": 3,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns both the generated answer and the relevant chunks used to generate it.
    """
    if vectorstore is None or STARTUP_ERRORS:
        return not_ready()

    try:
        params = read_params()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        answer, chunks = rag(vectorstore=vectorstore, **params)
    except Exception as e:
        log.error("Query failed:\n%s", traceback.format_exc())   # full traceback in `docker logs broker-backend`
        return jsonify({"error": f"Unable to generate an answer: {type(e).__name__}: {e}"}), 500

    return jsonify({**params, "relevant_chunks": chunks, "answer": answer})


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """
    Handles POST requests to /v1/answer.

    Accepts the same request body as /v1/answer_with_relevant_chunks
    and returns only the generated answer.
    """
    if vectorstore is None or STARTUP_ERRORS:
        return not_ready()

    try:
        params = read_params()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        generated_answer, _ = rag(vectorstore=vectorstore, **params)
    except Exception as e:
        log.error("Query failed:\n%s", traceback.format_exc())
        return jsonify({"error": f"Unable to generate an answer: {type(e).__name__}: {e}"}), 500

    return jsonify({"answer": generated_answer})


# Run the Flask application if this script is executed directly
if __name__ == "__main__":
    rag_api.run(host="0.0.0.0", port=7860)
