from langchain_core.tools import tool

_search_counter = 0

def reset_search_counter():
    global _search_counter
    _search_counter = 0

@tool
def web_search(query: str) -> str:
    """Search the web for real-time information."""
    return f"Web search results for: '{query}'"

@tool
def document_search(query: str) -> str:
    """Search internal knowledge base documents for relevant DSA context."""
    return f"Document search results for: '{query}'"
