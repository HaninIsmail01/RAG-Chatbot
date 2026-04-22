from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#required imports
import chainlit as cl
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

from src.RAG.agent_graph import build_chain, RAGChain
from src.RAG.chat_history import SessionMemory
from config.logging import get_logger

logger = get_logger(__name__)


@cl.on_chat_start #decorator to run this function at the start of each chat session 
async def on_chat_start():
    """
    Called once per session when a user opens the chat.
    Initialises the RAG chain and session memory, stores
    them in the Chainlit session so they persist across turns.
    
    Also sends a welcome message to the user and logs the session start.
        
    Returns:
        None
    """
    logger.info("New chat session started")

    await cl.Message(
        content=(
            "👋 Hello! I'm your Iphone guide assistant.\n\n"
            "Ask me anything about your iphone and I'll guide with cited answers from the user manual! 📱📖"
        )
    ).send()

    chain = build_chain() #build the RAG chain (retriever + graph) and store in session
    memory = SessionMemory() #initialize empty session memory to track conversation history and context

    #store the chain and memory in the session so they can be accessed in on_message for each user query
    cl.user_session.set("chain", chain) 
    cl.user_session.set("memory", memory) 

    logger.info("Chain and memory initialised for session")


@cl.on_message
async def on_message(message: cl.Message): 
    """
    Called on every user message.
    Streams the RAG response token by token and appends citations.
    Steps:
    1. Retrieve the RAG chain and memory from the session
    2. Stream the RAG response token by token to the UI for a dynamic experience
    3. Handle any errors gracefully and log them for debugging
    
    Args:
        message (cl.Message): The user message.
        
    Returns:
        None
    """
    chain: RAGChain = cl.user_session.get("chain")
    memory: SessionMemory = cl.user_session.get("memory")

    if not chain or not memory: 
        #check if chain and memory have been initialised in on_chat_start
        await cl.Message(
            content="Session error — please refresh the page."
        ).send()
        return

    # Show a thinking indicator while retrieving + generating
    async with cl.Step(name="Retrieving from document..."):
        pass

    # Stream the response
    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        response = await chain.ainvoke( 
            #run the RAG chain
            query=message.content,
            memory=memory,
        )
        response_msg.content = response
        await response_msg.update()

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        response_msg.content = (
            "An error occurred while processing your question. "
            "Please try again."
        )
        await response_msg.update()