"""
CLI query interface — ask a single question from the command line.

Usage:
    python query.py "What is the estimated value of 742 Evergreen Terrace?"
"""

import logging
import sys

import config
from chatbot.chatbot import PropertyValuationChatbot
from demo.cli_demo import print_answer, print_header, print_retrieved_docs

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python query.py \"Your question here\"")
        print("Example: python query.py \"What is the estimated value of 742 Evergreen Terrace?\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print_header()
    chatbot = PropertyValuationChatbot()

    from rich.console import Console
    console = Console()

    with console.status("[bold green]Searching & generating..."):
        response = chatbot.ask(query)

    print_retrieved_docs(response)
    print_answer(response)


if __name__ == "__main__":
    main()
