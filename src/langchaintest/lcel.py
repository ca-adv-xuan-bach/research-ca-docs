from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter API key for Azure: ")


model = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_API_BASE_URL"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {topic}."),
    ("human", "{query}")
])

chain = prompt | model | StrOutputParser()

# Run the chain
response = chain.invoke({
    "topic": "artificial intelligence",
    "query": "Explain the benefits of using LangChain"
})


print(response)
