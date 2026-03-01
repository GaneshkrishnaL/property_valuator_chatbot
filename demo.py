"""
Pre-scripted demo for the Google interview presentation.

Runs 5 queries that each demonstrate a different capability:
  1. Simple valuation → basic RAG retrieval
  2. Structured filtering → hybrid search with metadata
  3. Comparison → multi-document retrieval
  4. Market analysis → cross-document synthesis
  5. Edge case → hallucination prevention (nonexistent property)

Usage:
    python demo.py           # Run all demo queries
    python demo.py --interactive  # Interactive chat mode
"""

import argparse
import logging
import sys

import config
from chatbot.chatbot import PropertyValuationChatbot
from demo.cli_demo import run_demo_queries, run_interactive

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── Demo queries — each demonstrates a different capability ──
DEMO_QUERIES = [
    # Query 1: Simple valuation — demonstrates basic RAG
    "What is the estimated value of the property at 742 Evergreen Terrace, Austin?",

    # Query 2: Structured filtering — demonstrates hybrid search with metadata
    "What can you tell me about 3-bedroom homes in Downtown Austin priced under $600,000?",

    # Query 3: Comparison — demonstrates multi-document retrieval
    "Compare the property at 742 Evergreen Terrace with similar homes in the same neighborhood",

    # Query 4: Market analysis — demonstrates cross-document synthesis
    "What are the current market trends in the Austin real estate market? Are prices going up or down?",

    # Query 5: Edge case — demonstrates hallucination prevention
    "What is the property value at 999 Nonexistent Street, Miami?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Property Valuation RAG Demo")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive chat mode")
    args = parser.parse_args()

    chatbot = PropertyValuationChatbot()

    if args.interactive:
        run_interactive(chatbot)
    else:
        run_demo_queries(chatbot, DEMO_QUERIES)


if __name__ == "__main__":
    main()
