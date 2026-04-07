from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question (str): The user's question.
        generation (str): The LLM-generated answer.
        web_search (str): Whether to add web search.
        documents (list): The documents retrieved from the vector store.
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]