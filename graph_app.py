import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages
from langchain_groq import ChatGroq
from langchain_mcp_tools import convert_mcp_to_langchain_tools

load_dotenv()


async def build_graph():
    api_token = os.getenv("BRIGHT_DATA_API_KEY")
    if not api_token:
        raise RuntimeError("BRIGHT_DATA_API_KEY is not set")

    mcp_servers = {
        "bright_data": {
            "url": f"https://mcp.brightdata.com/sse?token={api_token}",
            "transport": "sse",
        }
    }

    # This is the part that used to work for you
    tools, cleanup = await convert_mcp_to_langchain_tools(mcp_servers)

    # Optional filter if you only want specific tools
    allowed_tool_names = {"search_engine"}
    data_tools = [t for t in tools if t.name in allowed_tool_names]

    groq_model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.15,
        max_tokens=450,
    )

    # Agent 1: pick 2 liquid US stocks
    stock_finder_agent = create_react_agent(
        groq_model,
        tools=[],
        prompt=(
            "You are a stock picker focused on liquid US equities on NASDAQ and NYSE.\n"
            "Input is user free text that may include an explicit intent line:\n"
            "\"User intent: BUY\" or \"User intent: SELL\".\n"
            "\n"
            "Rules:\n"
            "- Always pick exactly 2 stocks.\n"
            "- Only US stocks listed on NASDAQ or NYSE.\n"
            "- Avoid penny stocks and illiquid names.\n"
            "- Respect the user intent:\n"
            "  - If intent is BUY, both picks must be buy ideas.\n"
            "  - If intent is SELL, both picks must be sell or hold ideas.\n"
            "\n"
            "Output format (plain text, not JSON):\n"
            "1) NAME (TICKER) - one short reason\n"
            "2) NAME (TICKER) - one short reason\n"
        ),
        name="stock_finder_agent",
    )

    # Agent 2: fetch market data using MCP tools
    market_data_agent = create_react_agent(
        groq_model,
        tools=data_tools,
        prompt=(
            "You enrich information with recent US market data for the 2 tickers.\n"
            "Use tools when available to fetch data, especially the search_engine tool.\n"
            "For each ticker, find at least:\n"
            "- current price\n"
            "- previous close\n"
            "- volume\n"
            "- very short trend description such as Upward, Downward, Sideways\n"
            "\n"
            "Return a short text block per stock without extra formatting.\n"
        ),
        name="market_data_agent",
    )

    # Agent 3: sentiment and news summary
    news_analyst_agent = create_react_agent(
        groq_model,
        tools=data_tools,
        prompt=(
            "You analyze very recent news for each ticker.\n"
            "Use tools when available to search for news, especially the search_engine tool.\n"
            "\n"
            "For each stock, summarize:\n"
            "- one or two key recent news items\n"
            "- sentiment as one of: Positive, Negative, Neutral\n"
            "\n"
            "Return concise text, one block per stock.\n"
        ),
        name="news_analyst_agent",
    )

    # Agent 4: final structured JSON recommendation
    price_recommender_agent = create_react_agent(
        groq_model,
        tools=[],
        prompt=(
            "You receive the full conversation history from other agents.\n"
            "Your task is to produce final structured output for exactly 2 US stocks.\n"
            "\n"
            "User intent rules:\n"
            "- If the initial user message contains \"User intent: BUY\" then both stocks must have action \"BUY\".\n"
            "- If the initial user message contains \"User intent: SELL\" then each action must be either \"SELL\" or \"HOLD\".\n"
            "- If no explicit intent is present, default to BUY actions.\n"
            "\n"
            "For each stock, use the best information from the previous agents.\n"
            "Populate fields even if some numbers are approximate.\n"
            "\n"
            "Output format:\n"
            "{\n"
            "  \"stocks\": [\n"
            "    {\n"
            "      \"name\": \"Full company name\",\n"
            "      \"ticker\": \"TICKER\",\n"
            "      \"action\": \"BUY\" or \"SELL\" or \"HOLD\",\n"
            "      \"targetPrice\": \"$123.45\",\n"
            "      \"currentPrice\": \"$120.00\",\n"
            "      \"trend\": \"Upward\" or \"Downward\" or \"Sideways\",\n"
            "      \"sentiment\": \"Positive\" or \"Negative\" or \"Neutral\",\n"
            "      \"reason\": \"One or two short sentences explaining the recommendation\"\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"...\",\n"
            "      \"ticker\": \"...\",\n"
            "      \"action\": \"...\",\n"
            "      \"targetPrice\": \"$...\",\n"
            "      \"currentPrice\": \"$...\",\n"
            "      \"trend\": \"...\",\n"
            "      \"sentiment\": \"...\",\n"
            "      \"reason\": \"...\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "\n"
            "Return only the JSON. Do not include explanations outside the JSON.\n"
        ),
        name="price_recommender_agent",
    )

    supervisor_graph = create_supervisor(
        model=groq_model,
        agents=[
            stock_finder_agent,
            market_data_agent,
            news_analyst_agent,
            price_recommender_agent,
        ],
        prompt=(
            "You orchestrate 4 agents to answer the user.\n"
            "Call them in this strict order when appropriate:\n"
            "1) stock_finder_agent\n"
            "2) market_data_agent\n"
            "3) news_analyst_agent\n"
            "4) price_recommender_agent\n"
            "\n"
            "Stop after price_recommender_agent returns the final JSON.\n"
            "Do not call agents unnecessarily.\n"
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()

    # cleanup is async and will be awaited by FastAPI lifespan
    return supervisor_graph, cleanup


def get_history_from_output(graph_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the messages list from the graph output in a consistent way.
    """
    if "supervisor" in graph_output and isinstance(graph_output["supervisor"], dict):
        return graph_output["supervisor"].get("messages", [])
    return graph_output.get("messages", [])


def extract_final_text(graph_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the final JSON object from the last message of the supervisor history.
    If parsing fails or the format is unexpected, return an empty structure.
    """
    history = get_history_from_output(graph_output)
    messages = convert_to_messages(history)

    if not messages:
        return {"stocks": []}

    last = messages[-1]
    if hasattr(last, "content"):
        text = last.content
    else:
        text = str(last)

    if not isinstance(text, str):
        text = str(text)

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "stocks" in data:
            return data
    except json.JSONDecodeError:
        pass

    return {"stocks": []}


def build_timeline_with_result(graph_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a simple timeline array plus the final result dict.
    The API layer turns this into Pydantic models.
    """
    history = get_history_from_output(graph_output)
    messages = convert_to_messages(history)

    timeline: List[Dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = getattr(msg, "role", getattr(msg, "type", "")) or "ai"
        agent = getattr(msg, "name", None)
        if not agent:
            if role == "human":
                agent = "user"
            else:
                agent = "unknown"

        content = msg.content
        if not isinstance(content, str):
            content = str(content)

        if role == "human" and idx == 0:
            label = "User query"
        elif agent == "stock_finder_agent":
            label = "Stock selection"
        elif agent == "market_data_agent":
            label = "Market data fetch"
        elif agent == "news_analyst_agent":
            label = "News and sentiment"
        elif agent == "price_recommender_agent":
            label = "Final JSON"
        else:
            label = "Message"

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
