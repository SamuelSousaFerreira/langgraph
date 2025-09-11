from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send  

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
from tavily import TavilyClient
import datetime


from schemas import * 
from prompts import *


from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-pro")

#Nós
def build_first_queries(state: ReportState):
    class QueryList(BaseModel):
        queries: List[str]
    
    user_input = state.user_input
    prompt = build_queries.format(user_input=user_input)
    query_llm = llm.with_structured_output(QueryList)
    result = query_llm.invoke({"input": prompt})
      
    return {"queries": result.queries}
#


def single_search(query:str):
    tavily_client = TavilyClient()
    
    results = tavily_client.search(query,
                                   max_results=1,
                                   include_raw_content=False)
    url = results['results'][0]['url']
    url_extraction = tavily_client.extract(url)
    
    if len(url_extraction['results']) > 0:
        raw_content = url_extraction["results"][0]["raw_content"]
        prompt = resume_search.format(user_input=user_input, search_result=raw_content)
        llm_result = llm.invoke({prompt})
        query_results = QueryResult(
            title = title=results
        )



#Edges
builder = StateGraph(ReportState)

graph = builder.compile()

if __name__ == "__main__":
    
    user_input = """ How create a LLM from scratch?"""
    graph.invoke({"user_input": user_input})