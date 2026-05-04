from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generated answers."""
    
    binary_score: bool = Field(description="Answer is grounded in the facts, True or False.")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system="""
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved documents.
Give a binary score of True or False to indicate whether the answer is grounded in the facts.
"""

hallucination_prommpt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts:\n\n{documents}\n\nLLM generation:\n{generation}"),
    ]
)


hallucination_grader = hallucination_prommpt | structured_llm_grader