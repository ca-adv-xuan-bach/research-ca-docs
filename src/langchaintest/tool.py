from langchain_core.messages import HumanMessage, SystemMessage
from langchain.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType


from langchain_openai import AzureChatOpenAI
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter API key for Azure: ")

if not os.environ.get("SERPAPI_API_KEY"):
    os.environ["SERPAPI_API_KEY"] = getpass.getpass(
        "Enter API key serp: ")

search = SerpAPIWrapper(serpapi_api_key=os.environ["SERPAPI_API_KEY"])
search_tool = Tool(
    name="Google Search",
    func=search.run,
    description="find on google"
)

model = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_API_BASE_URL"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

agent = initialize_agent(
    tools=[search_tool],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.run("who is current america president")
print(response)
