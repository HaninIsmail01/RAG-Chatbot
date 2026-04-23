from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chainlit as cl
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

from src.RAG.agent_graph import build_chain, RAGChain
from src.RAG.chat_history import SessionMemory
from config.logging import get_logger

logger = get_logger(__name__)


@cl.on_chat_start
async def on_chat_start():
    logger.info("New chat session started")

    chain = build_chain()
    memory = SessionMemory()

    cl.user_session.set("chain", chain)
    cl.user_session.set("memory", memory)

    await cl.Message(
        content=(
            "👋 Hello! I'm your Iphone guide.\n\n"
            "Ask me anything about your iphone, and I'll answer "
        ),
        author="Assistant",
    ).send()

    logger.info("Chain and memory initialised for session")


@cl.on_message
async def on_message(message: cl.Message):
    chain: RAGChain = cl.user_session.get("chain")
    memory: SessionMemory = cl.user_session.get("memory")

    if not chain or not memory:
        await cl.Message(
            content="⚠️ Session error — please refresh the page.",
            author="Assistant",
        ).send()
        return

    # Create the response message and send it immediately
    # so Chainlit registers it as a new message in the thread
    response_msg = cl.Message(content="", author="Assistant")
    await response_msg.send()

    try:
        async for token in chain.astream(
            query=message.content,
            memory=memory,
        ):
            await response_msg.stream_token(token)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        response_msg.content = (
            "⚠️ An error occurred while processing your question. "
            "Please try again."
        )
        await response_msg.update()