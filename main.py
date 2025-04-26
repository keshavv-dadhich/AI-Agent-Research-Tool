from workflow import build_workflow

research_flow = build_workflow()

def run_research(query: str):
    results = research_flow.invoke({
        "input": query,
        "research_data": [],
        "answer": ""
    })
    return results["answer"]

if __name__ == "__main__":
    print(run_research("Latest AI agent architectures in 2024"))
