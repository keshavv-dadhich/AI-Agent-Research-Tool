from langchain.schema import SystemMessage
from langchain_anthropic import ChatAnthropic

class AnswerDrafter:
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-3-opus-20240229",
            temperature=0.3
        )
        
    def draft(self, research_data: str):
        return self.llm.invoke([
            SystemMessage(content="Synthesize this into a structured answer:"),
            research_data
        ]).content
