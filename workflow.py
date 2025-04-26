# workflow.py
from langgraph.graph import StateGraph, END  # Fixed import
from typing import TypedDict, Annotated, List
from agents.research_agent import ResearchAgent
from agents.answer_drafter import AnswerDrafter

class ResearchState(TypedDict):
    input: str
    research_data: Annotated[List[str], lambda x, _: x.extend(_)]
    answer: str

def build_workflow():
    workflow = StateGraph(ResearchState)
    
    # Initialize agents
    researcher = ResearchAgent().create()
    drafter = AnswerDrafter()
    
    # Define nodes
    def research_node(state):
        result = researcher.invoke({"input": state["input"]})
        return {"research_data": [result["output"]]}
    
    def draft_node(state):
        answer = drafter.draft("\n".join(state["research_data"]))
        return {"answer": answer}
    
    # Build graph
    workflow.add_node("research", research_node)
    workflow.add_node("draft", draft_node)
    
    workflow.set_entry_point("research")
    workflow.add_edge("research", "draft")
    workflow.add_edge("draft", END)  # Now works with imported END
    
    return workflow.compile()
