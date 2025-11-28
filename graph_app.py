# graph_app.py
import os
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages
from langchain_mcp_tools import convert_mcp_to_langchain_tools
from langchain_groq import ChatGroq

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

    tools, cleanup = await convert_mcp_to_langchain_tools(mcp_servers)

    # Only the market data agent will see this tool
    allowed_tool_names = {"search_engine"}
    data_tools = [t for t in tools if t.name in allowed_tool_names]

    model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.2,
        max_tokens=300,
    )

    # Agent 1: pick 2 stocks
    stock_finder_agent = create_react_agent(
        model,
        tools=[],
        prompt=(
            "You pick exactly 2 NSE listed stocks for short term trading. "
            "Avoid penny and illiquid stocks. "
            "Return name, ticker, and one short reason for each."
        ),
        name="stock_finder_agent",
    )

    # Agent 2: fetch market data using MCP
    market_data_agent = create_react_agent(
        model,
        tools=data_tools,
        prompt=(
            "You fetch current market data for the 2 chosen NSE stocks. "
            "Use the search_engine tool when you need recent data. "
            "For each stock, return current price in INR, rough 7 to 30 day trend, "
            "today volume, and whether volume looks high, normal, or low. "
            "Be concise."
        ),
        name="market_data_agent",
    )

    # Agent 3: sentiment and news summary (no tools)
    news_analyst_agent = create_react_agent(
        model,
        tools=[],
        prompt=(
            "You summarize recent news and sentiment for each stock using the conversation context. "
            "If news is not explicit, give a neutral summary. "
            "For each stock, return 1 to 2 lines and label sentiment as positive, negative, or neutral."
        ),
        name="news_analyst_agent",
    )

    # Agent 4: final trading call (no tools)
    price_recommender_agent = create_react_agent(
        model,
        tools=[],
        prompt=(
            "You give a near term trading call for each stock based on previous messages. "
            "For each stock, choose Buy, Sell, or Hold, a short term target price in INR, "
            "and a 2 to 3 line justification. Be practical and concise."
        ),
        name="price_recommender_agent",
    )

    workflow = create_supervisor(
        agents=[
            stock_finder_agent,
            market_data_agent,
            news_analyst_agent,
            price_recommender_agent,
        ],
        model=model,
        prompt=(
            "You supervise four agents:\n"
            "- stock_finder_agent picks 2 NSE stocks.\n"
            "- market_data_agent fetches market data with tools.\n"
            "- news_analyst_agent summarizes sentiment.\n"
            "- price_recommender_agent gives the final trading call.\n"
            "Call each agent at most once and in a logical order. "
            "Do not loop agents. "
            "At the end, answer the user with:\n"
            "1) The 2 chosen stocks (name and ticker).\n"
            "2) Short market data summary per stock.\n"
            "3) Short sentiment summary per stock.\n"
            "4) A clear Buy, Sell, or Hold with target price per stock.\n"
            "Do not mention agents or tools."
        ),
        add_handoff_back_messages=True,
        output_mode="last_message",
    )

    app = workflow.compile()
    return app, cleanup


def extract_final_text(graph_output: dict) -> str:
    messages = convert_to_messages(graph_output["messages"])
    last = messages[-1]
    if hasattr(last, "content"):
        return last.content
    return str(last)
