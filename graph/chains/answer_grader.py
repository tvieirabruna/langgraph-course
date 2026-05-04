from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class GradeHAnswer(BaseModel):
    """Binary score for whether the answer addresses the user's question."""
    
    binary_score: bool = Field(description="Answer addresses the user's question, True or False.")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHAnswer)

system="""
You are a grader assessing whether an LLM generation addresses / resolves the user's question.
Give a binary score of True or False to indicate whether the answer addresses / resolves the user's question.
"""

answer_grader_prommpt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question:\n\n{question}\n\nLLM generation:\n{generation}"),
    ]
)


answer_grader = answer_grader_prommpt | structured_llm_grader