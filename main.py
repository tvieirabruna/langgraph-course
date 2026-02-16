from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import reflection_chain, generate_chain

from dotenv import load_dotenv

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph) -> dict:
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph) -> dict:
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: MessageGraph) -> str:
    if len(state["messages"]) >= 6:
        return END
    return REFLECT
    

builder = StateGraph(state_schema=MessageGraph)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)

builder.add_conditional_edges(GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="flow.png")
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()


def main():
    print("Hello, Reflection Agent!")
    inputs = {
        "messages": [
            HumanMessage(content="""Make this tweet better:
                          @LangChainAI
                          - newly Tool Calling feature is serioulsy underrated.
                          
                          After a long wait, it's here - making the implementation of agents accross different models with function calling.
                          
                          Made a video covering their newest blog post.
                          """)
        ]
    }
    res = graph.invoke(inputs)
    print(res)


if __name__ == "__main__":
    main()
