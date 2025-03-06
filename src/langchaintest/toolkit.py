from langchain_core.messages import HumanMessage, SystemMessage
from langchain.utilities import SerpAPIWrapper
from langchain.tools import Tool, StructuredTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


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
    name="GoogleSearch",
    description="Useful for searching Google to find information about current events, facts, or up-to-date information.",
    func=search.run
)


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b


calculator_tool = StructuredTool.from_function(
    func=add_numbers,
    name="Calculator",
    description="Useful for adding two numbers together. Input should be two numbers."
)

tools = [search_tool, calculator_tool]

model = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_API_BASE_URL"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with two tools: a search tool to find information online, and a calculator tool to add two numbers together. Choose the appropriate tool based on the user's request."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

queries = [
    "What is the capital of Vietnam?",
    "Can you add 123.45 and 678.9 for me?"
]

for query in queries:
    print(f"\n\nQuery: {query}")
    result = agent_executor.invoke({"input": query})
    print(f"Result: {result['output']}\n")
    print("-" * 50)
