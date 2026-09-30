from langchain_openai import ChatOpenAI


# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 4, vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]


# Generate an answer by prompting the LLM with the retrieved context
def generate(query, retrieved_chunks, model):

    prompt = f"""
    You are an AI assistant helping GlobalEdge Brokerage brokers answer questions using
    financial news, stock-price data and SEC filings.

    User Query:
    {query}

    Retrieved Evidence:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved evidence.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved evidence.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important figures, dates, conditions, risks, or caveats that appear in the evidence.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved evidence.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent prices, percentages, dates, company events, or filing details.
    10. If the information is insufficient, say:
        "The available data does not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)


# Full RAG pipeline: retrieval followed by generation
def rag(
    query: str,
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model (reads OPENAI_API_KEY and OPENAI_API_BASE from the environment)
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)

    # Generate the answer from the retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
