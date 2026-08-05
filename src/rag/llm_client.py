"""
src/rag/llm_client.py

Wraps the Groq API call: takes a user query + ticker, retrieves relevant
context via retriever.py, assembles a grounded prompt, and returns the
generated answer. This is the function the Streamlit chat page calls
directly -- it should never need to know about FAISS, chunks, or Groq's
request format itself.
"""

import logging
import os

from dotenv import load_dotenv
from groq import Groq

from src.rag.retriever import retrieve_relevant_chunks

load_dotenv()
from src.utils.config import load_config as _load_config

logger = logging.getLogger("equirisk.rag.llm_client")

SYSTEM_PROMPT = """You are EquiRisk's assistant, helping a user understand the risk profile \
of a specific Nifty150 midcap stock. You are given retrieved context (recent news headlines \
and computed risk statistics) for the ticker the user is asking about.

Rules:
- Base your answer ONLY on the provided context. If the context doesn't contain enough \
information to answer, say so plainly rather than guessing.
- Do not give direct buy/sell/hold investment advice -- describe what the data and risk \
score indicate, and let the user draw their own conclusion.
- Be concise and specific -- cite the actual numbers/labels from the context rather than \
vague statements like "the stock seems risky."
"""


def _get_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _build_prompt(ticker: str, query: str, context_chunks: list) -> str:
    if not context_chunks:
        context_block = "No retrieved context available for this ticker."
    else:
        context_block = "\n".join(f"- {c}" for c in context_chunks)

    return (
        f"Ticker: {ticker}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"User question: {query}"
    )


def answer_query(ticker: str, query: str, config_path: str = None) -> str:
    """Main entrypoint called by the Streamlit chat page.

    Returns a plain string answer. If no index exists yet for the
    ticker (pipeline hasn't been run), returns a clear message telling
    the user to refresh the pipeline rather than silently hallucinating
    an answer.
    """
    config = _load_config(config_path)
    genai_config = config["genai"]

    context_chunks = retrieve_relevant_chunks(ticker, query, config_path)
    if not context_chunks:
        return (
            f"I don't have any indexed data for {ticker} yet. "
            f"Try clicking 'Refresh Pipeline' to fetch the latest data first."
        )

    prompt = _build_prompt(ticker, query, context_chunks)
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=genai_config["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=genai_config["temperature"],
            max_tokens=genai_config["max_tokens"],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return "Sorry, I couldn't generate a response right now -- please try again in a moment."


if __name__ == "__main__":
    answer = answer_query("TATAMOTORS", "why is this stock considered risky right now?")
    print(answer)