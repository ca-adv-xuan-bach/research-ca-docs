import getpass
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")


model = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_API_BASE_URL"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

message = model.invoke("Hello, world!")

print(message)
