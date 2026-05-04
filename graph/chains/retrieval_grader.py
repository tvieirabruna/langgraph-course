from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    
    binary_score: str = Field(description="Whether the documents are relevant to the user's question, 'yes' or 'no'.")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system="""
You are a grader assessing the relevance of a retrieved document to a user's question.
If the document contains keyword(s) or semantic meaning(s) related to the user's question, grade it as relevant.
Give a binary score of 'yes' or 'no' to indicate whether the document is relevant to the user's question.
"""

grade_prommpt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document:\n\n{document}\n\nUser question:\n{question}"),
])


retrieval_grader = grade_prommpt | structured_llm_grader