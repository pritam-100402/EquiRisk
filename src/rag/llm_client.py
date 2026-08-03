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

SYSTEM_PROMPT = """You are an equity risk analyst assistant for EquiRisk, a
dashboard covering the Nifty Midcap 150.

Answer strictly from the context provided. If the context does not contain the
answer, say so plainly rather than filling the gap from general knowledge.

When the user asks whether they should buy, sell, or invest in a stock, do NOT
give a recommendation. You are not a licensed adviser and this is an educational
project. Instead, lay out the evidence so the user can decide, structured as:

  **Risk profile** - the predicted risk category and what the volatility figures
  say about how much the price moves.

  **Fundamentals** - sector, market capitalisation, P/E, profitability and
  leverage, with a sentence on what each implies. Note where higher leverage
  tends to amplify volatility.

  **Recent news** - what the retrieved headlines suggest, and their overall tone.

  **What to consider** - the factors that would matter for this decision, framed
  as questions for the user rather than as advice.

Close with a brief reminder that this is educational analysis, not investment
advice, and that the risk label is a model prediction with a measurable error
rate rather than a certainty.

For all other questions, answer directly and concisely in plain prose. Quote
specific figures from the context wherever they are available. Use Indian
conventions - rupees, crores - when discussing amounts."""




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