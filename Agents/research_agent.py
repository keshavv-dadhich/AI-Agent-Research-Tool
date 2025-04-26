from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from tools.web_search import TavilyTools

from langgraph.prebuilt import ToolNode

class ResearchAgent:
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0
        )
        self.tools = TavilyTools().web_search
        
    def create(self):
        # Convert string prompt to ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a research specialist. Given: {input}"),
            ("human", "Always use web_search tool. Analyze and collect relevant data.")
        ])
        
        # Use RunnablePassthrough to maintain compatibility
        agent = (
            RunnablePassthrough.assign(
                input=lambda x: x["input"],
                scratchpad=lambda x: format_log_to_str(x["intermediate_steps"])
            )
            | prompt
            | self.llm.bind_tools([self.tools])
            | ToolNode([self.tools])
        )
        
        return AgentExecutor(agent=agent, tools=[self.tools])
