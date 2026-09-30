import os
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Base URL of the Flask backend. `broker-backend` is the backend container name on the shared Docker network.
BACKEND_URL = os.getenv("BACKEND_URL", "http://broker-backend:7860").rstrip("/")

# Backend endpoint used for each response type
ENDPOINTS = {
    "Answer only": "/v1/answer",
    "Answer + Relevant Chunks": "/v1/answer_with_relevant_chunks",
    "Relevant Chunks only": "/v1/relevant_chunks",
}


# ---------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------

st.set_page_config(page_title="Stock Broker Intelligence Assistant", page_icon="📈")


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def make_session():
    """HTTP session that retries only *connection* failures (backend still starting / network hiccup).
    Read timeouts are not retried, so a slow LLM call is never sent twice."""
    retry = Retry(total=None, connect=3, read=0, status=0, other=0, backoff_factor=2)
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def explain_connection_error(e):
    """Turn a low-level requests exception into an actionable hint."""
    text = str(e)
    if isinstance(e, requests.exceptions.ConnectTimeout) or "timed out" in text.lower():
        return (
            "The name was found but nothing answered in time. Usually the frontend and backend containers are NOT on the "
            "same Docker network (or the backend is not running). In the Codespace terminal run "
            "`docker network inspect broker-net` - both `broker-backend` and `broker-frontend` must be listed. "
            "If `broker-frontend` is missing: `docker network connect broker-net broker-frontend`."
        )
    if "Name or service not known" in text or "Failed to resolve" in text or "NameResolution" in text:
        return (
            "The backend name could not be resolved. Start both containers with `--network broker-net` and make sure "
            f"the backend container name matches the host in {BACKEND_URL}."
        )
    if "Connection refused" in text or "NewConnectionError" in text:
        return "The backend container is reachable but nothing is listening yet. Check `docker logs broker-backend`."
    return "Check `docker ps` and `docker logs broker-backend`."


def show_chunks(chunks):
    """Display each retrieved chunk in its own expander."""
    st.subheader("Relevant Chunks")
    for i, chunk in enumerate(chunks, start=1):
        with st.expander(f"Chunk {i}"):
            st.write(chunk)


session = make_session()


# ---------------------------------------------------------
# Application title
# ---------------------------------------------------------

st.title("Stock Broker Intelligence Assistant")
st.caption("Ask questions about financial news, stock prices and SEC filings. Answers are grounded in retrieved evidence.")


# ---------------------------------------------------------
# Sidebar: connection check so problems are visible before asking a question
# ---------------------------------------------------------

with st.sidebar:
    st.subheader("Backend status")
    st.code(BACKEND_URL)
    if st.button("Check connection"):
        try:
            r = session.get(f"{BACKEND_URL}/health", timeout=(10, 20))
            if r.status_code == 200:
                st.success(f"Connected. {r.json().get('vectors')} vectors loaded.")
            else:
                st.error(f"Backend replied HTTP {r.status_code}: {r.text[:500]}")
        except requests.exceptions.RequestException as e:
            st.error(f"Cannot reach backend: {type(e).__name__}")
            st.info(explain_connection_error(e))


# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------

response_type = st.radio("What would you like to see?", list(ENDPOINTS), horizontal=True)

query = st.text_input(
    "Ask your question",
    placeholder="e.g., What risk disclosures or financial discussions are mentioned in the SEC filings for Goldman Sachs?"
)

# Number of chunks retrieved from the vector database
k = st.slider("Number of evidence chunks to retrieve", min_value=1, max_value=10, value=4)


# ---------------------------------------------------------
# Submit button
# ---------------------------------------------------------

if st.button("Submit", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")
    else:
        response, conn_error = None, None
        with st.spinner("Retrieving evidence and generating the answer..."):
            try:
                response = session.post(
                    f"{BACKEND_URL}{ENDPOINTS[response_type]}",
                    json={"query": query, "k": k},
                    timeout=(15, 180),   # 15 s to connect (retried 3 times), 180 s to wait for the answer
                )
            except requests.exceptions.RequestException as e:
                conn_error = e

        if conn_error is not None:
            st.error(f"Unable to connect to the RAG API at {BACKEND_URL}.")
            st.code(f"{type(conn_error).__name__}: {conn_error}")
            st.info(explain_connection_error(conn_error))

        elif response.status_code == 200:
            result = response.json()

            if response_type in ("Answer only", "Answer + Relevant Chunks"):
                st.subheader("Answer")
                st.write(result["answer"])

            if response_type in ("Answer + Relevant Chunks", "Relevant Chunks only"):
                show_chunks(result["relevant_chunks"])

        else:
            try:
                body = response.json()
                detail = body.get("error", response.text)
                if body.get("details"):
                    detail += " | " + "; ".join(body["details"])
            except ValueError:
                detail = response.text
            st.error(f"The RAG API returned HTTP {response.status_code}: {detail}")
