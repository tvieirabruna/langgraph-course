from dotenv import load_dotenv
from pprint import pprint

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from ingestion import retriever

load_dotenv()


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    
    assert res.binary_score == "yes"
    

def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizza", "document": doc_txt}
    )
    
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
    
    
def test_hallucination_grader_answer_true() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    
    generation = generation_chain.invoke({"context": docs, "question": question})    
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    
    assert res.binary_score == True


def test_hallucination_grader_answer_false() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    
    generation = generation_chain.invoke({"context": docs, "question": question})    
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": "In order to make a pizza, you need to have a pizza oven and a pizza dough."}
    )
    
    assert res.binary_score == False