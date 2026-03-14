# core/memory.py

from typing import List, Tuple

# Max number of conversation turns to keep in memory.
# Each turn = 1 human message + 1 AI response.
# Higher = more context but larger prompts = more tokens used.
MAX_HISTORY_TURNS = 5


class ConversationMemory:
    """
    Simple in-memory conversation history manager.

    Stores (human_message, ai_message) pairs.
    Automatically trims to MAX_HISTORY_TURNS to avoid
    prompt getting too long.

    Why not use LangChain's built-in memory?
    Because explicit control is clearer for learning,
    and easier to persist to disk later.
    """

    def __init__(self):
        self.history: List[Tuple[str, str]] = []

    def add_turn(self, human_message: str, ai_message: str) -> None:
        """Add a completed conversation turn to memory."""
        self.history.append((human_message, ai_message))
        # Keep only the last N turns to control prompt size
        if len(self.history) > MAX_HISTORY_TURNS:
            self.history = self.history[-MAX_HISTORY_TURNS:]

    def get_history(self) -> List[Tuple[str, str]]:
        """Return full conversation history."""
        return self.history

    def clear(self) -> None:
        """Reset memory — useful when user uploads new documents."""
        self.history = []
        print("[Memory] Conversation history cleared.")

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        return f"ConversationMemory({len(self.history)} turns)"