import os
import json
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import convert_to_messages
from langchain_groq import ChatGroq

load_dotenv()


async def build_graph():
    model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.1,
        max_tokens=260,
    )

    stock_agent = create_react_agent(
        model,
        tools=[],
        prompt=(
            "You are a US stock picker focusing on NASDAQ and NYSE stocks.\n"
            "The user message will contain a line like:\n"
            "User intent: BUY\n"
            "or\n"
            "User intent: SELL\n"
            "Rules for action:\n"
            "- If intent is BUY, action must be BUY for both stocks.\n"
            "- If intent is SELL, action must be SELL or HOLD.\n"
            "Choose exactly 2 US listed stocks (NASDAQ or NYSE). Avoid penny and illiquid stocks.\n"
            "Keep every field short. The reason must be at most 12 words.\n"
            "Respond only with valid JSON in this format:\n"
            "{\n"
            "  \"stocks\": [\n"
            "    {\n"
            "      \"name\": \"...\",\n"
            "      \"ticker\": \"...\",\n"
            "      \"action\": \"BUY\",\n"
            "      \"targetPrice\": \"$1234\",\n"
            "      \"currentPrice\": \"$1200\",\n"
            "      \"trend\": \"Upward\",\n"
            "      \"sentiment\": \"Positive\",\n"
            "      \"reason\": \"...\"\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"...\",\n"
            "      \"ticker\": \"...\",\n"
            "      \"action\": \"BUY\",\n"
            "      \"targetPrice\": \"$...\",\n"
            "      \"currentPrice\": \"$...\",\n"
            "      \"trend\": \"Sideways\",\n"
            "      \"sentiment\": \"Neutral\",\n"
            "      \"reason\": \"...\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Do not output any text before or after the JSON.\n"
        ),
        name="stock_reco_agent",
    )

    app = stock_agent

    async def cleanup():
        return

    return app, cleanup


def extract_final_text(graph_output: dict) -> dict:
    messages = convert_to_messages(graph_output["messages"])
    last = messages[-1]

    if hasattr(last, "content"):
        text = last.content
    else:
        text = str(last)

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "stocks" in data:
            return data
    except json.JSONDecodeError:
        pass

    return {"stocks": []}


def build_timeline_with_result(graph_output: dict) -> dict:
    messages = convert_to_messages(graph_output.get("messages", []))
    timeline = []

    for idx, msg in enumerate(messages):
        role = getattr(msg, "role", getattr(msg, "type", "")) or "ai"
        agent = getattr(msg, "name", None)
        if not agent:
            agent = "user" if role == "human" else "stock_reco_agent"

        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if role == "human" and idx == 0:
            label = "User query"
        else:
            label = "Model turn"

        timeline.append(
            {
                "step": idx,
                "role": role,
                "agent": agent,
                "content": content,
                "label": label,
            }
        )

    final_result = extract_final_text(graph_output)

    return {
        "timeline": timeline,
        "final": final_result,
    }
