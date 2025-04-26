from tavily import TavilyClient
from langchain.tools import tool

class TavilyTools:
    def __init__(self):
        self.client = TavilyClient(api_key="")

    @tool
    def web_search(self, query: str) -> list:
        """Perform deep web search using Tavily"""
        return self.client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            max_results=5
        ).get('results', [])

    @tool
    def crawl_website(self, url: str) -> str:
        """Crawl specific website content"""
        return self.client.crawl(
            url=url,
            max_depth=2,
            max_breadth=5
        ).get('content', '')
