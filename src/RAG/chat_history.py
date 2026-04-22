#Implement memory for multiturn conversation support.
#The RAG graph needs access to the conversation history to provide context for retrieval and response generation.

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage


class SessionMemory:
    """
    Per-session in-memory chat history.

    Chainlit creates a new session per browser connection — this class
    holds the conversation history for that session so the RAG chain
    can reference prior turns for context.
    The history is stored in-memory and will be lost when the session ends.
    
    """

    def __init__(self):
        # The in-memory chat history provided by LangChain
        self._history = InMemoryChatMessageHistory()

    def add_user_message(self, message: str) -> None:
        # Add the user's message to the chat history
        self._history.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        # Add the assistant's response to the chat history
        self._history.add_ai_message(message)

    def get_messages(self) -> list[BaseMessage]:
        # Get all messages from the session history
        return self._history.messages

    def clear(self) -> None:
        # Clear the session history
        self._history.clear()