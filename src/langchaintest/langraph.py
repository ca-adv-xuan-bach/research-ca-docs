import getpass
import os
from typing import List, TypedDict

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")


model = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_API_BASE_URL"],
    azure_deployment=os.environ["OPENAI_MODEL_NAME"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
)


class ChatState(TypedDict):
    messages: List
    route: str


@tool
def get_weather(location: str) -> str:
    """Lấy thông tin thời tiết cho một địa điểm."""
    weather_data = {
        "Hà Nội": "32°C, nắng",
        "TP.HCM": "30°C, mưa rào",
    }
    return weather_data.get(location, f"Không có dữ liệu cho {location}")


agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn là trợ lý thông tin thời tiết. Sử dụng công cụ để lấy thông tin chính xác.",
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(model, [get_weather], agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=[get_weather])


# node 1
def router(state: ChatState) -> ChatState:
    last_message = state["messages"][-1].content.lower()

    if "thời tiết" in last_message:
        return {"messages": state["messages"], "route": "weather_agent"}
    else:
        return {"messages": state["messages"], "route": "normal_chat"}


# node 2
def weather_agent(state: ChatState) -> ChatState:
    messages = state["messages"]
    last_message = messages[-1]

    result = agent_executor.invoke({"messages": [last_message]})

    response = AIMessage(content=result["output"])

    return {"messages": messages + [response], "route": "end"}


# node 3
def normal_chat(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = AIMessage(
        content="Tôi là trợ lý đơn giản. Tôi có thể trả lời về thời tiết nếu bạn hỏi."
    )

    return {"messages": messages + [response], "route": "end"}


workflow = StateGraph(ChatState)
workflow.add_node("router", router)
workflow.add_node("weather_agent", weather_agent)
workflow.add_node("normal_chat", normal_chat)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda x: x["route"],
    {
        "weather_agent": "weather_agent",
        "normal_chat": "normal_chat",
    },
)
workflow.add_edge("weather_agent", END)
workflow.add_edge("normal_chat", END)
app = workflow.compile()

weather_query = HumanMessage(content="Thời tiết ở Hanoi hôm nay thế nào?")
result = app.invoke({"messages": [weather_query], "route": ""})
print(f"User: {result['messages'][0].content}")
print(f"Bot: {result['messages'][1].content}\n")

regular_query = HumanMessage(content="Bạn là ai?")
result = app.invoke({"messages": [regular_query], "route": ""})
print(f"User: {result['messages'][0].content}")
print(f"Bot: {result['messages'][1].content}")
