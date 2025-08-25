from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import datetime

from numpy import std

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

search_tool = TavilySearchResults(search_depth='basic')

@tool
def get_system_time(format: std = "%Y-%m-%d %H:%M:%S"):
    """Get the current system time in the specified format."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [search_tool, get_system_time]

agent = initialize_agent(tools=tools, 
                         llm=llm,
                         agent='zero-shot-react-description',
                         handle_parsing_errors=True,
                         verbose=True)

#agent.invoke("Give me a funny affirmation about Jacareí- SP")
# agent.invoke("Suppose that you birth on 1987-10-09. How Many years are you old?")
agent.invoke("Which was the result of the Sinquefield Cup 2025 today games?")

# result = llm.invoke("Give me a affirmation about Jacareí- SP")

# print(result.content)
