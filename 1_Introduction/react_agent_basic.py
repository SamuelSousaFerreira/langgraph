from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
#from langchain.agents import initialize_agent, tool
#from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = llm.invoke("Give me a fact about chess")

print(result.content)
