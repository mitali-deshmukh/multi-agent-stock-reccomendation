import os
import asyncio
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages
from langchain_mcp_tools import convert_mcp_to_langchain_tools
from langchain_groq import ChatGroq

load_dotenv()


def pretty_print_message(message, indent: bool = False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


async def run_agent(user_query: str):
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

    # Only keep the lightest tool and only use it in one agent
    allowed_tool_names = {"search_engine"}
    data_tools = [t for t in tools if t.name in allowed_tool_names]

    print("Loaded tools for data agent:", [t.name for t in data_tools])

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=800,
    )

    # Agent 1: finds candidate stocks 
    stock_finder_agent = create_react_agent(
        model,
        tools=[],
        prompt=(
            "You are an NSE stock picker. "
            "Given the user query and prior messages, pick exactly 2 NSE listed stocks "
            "for short term trading (a few days to a few weeks). "
            "Avoid penny stocks and very illiquid names. "
            "Output only their names and tickers with 1 line of reasoning each."
        ),
        name="stock_finder_agent",
    )

    # Agent 2: fetches data from web using Bright Data 
    market_data_agent = create_react_agent(
        model,
        tools=data_tools,
        prompt=(
            "You are a market data agent for NSE stocks. "
            "You can call the search_engine tool to look up current or recent data. "
            "For the 2 stocks mentioned in the conversation, fetch:\n"
            "- Current price\n"
            "- Recent price trend (last 7 to 30 days, high level)\n"
            "- Today volume and whether it is unusually high or low\n"
            "Return a compact summary for each stock. Use INR. "
            "Keep the text short."
        ),
        name="market_data_agent",
    )

    # Agent 3: news and sentiment (reasoning only, no tools to save tokens)
    news_analyst_agent = create_react_agent(
        model,
        tools=[],
        prompt=(
            "You are a news and sentiment analyst. "
            "Based on the conversation so far and any market data already described, "
            "summarize recent news themes or likely narratives for each of the 2 stocks. "
            "If news is not explicitly available, infer a plausible neutral summary instead of calling tools. "
            "Classify sentiment for each stock as positive, negative, or neutral and explain in 2 short lines."
        ),
        name="news_analyst_agent",
    )

    # Agent 4: makes final trading call (no tools)
    price_recommender_agent = create_react_agent(
        model,
        tools=[],
        prompt=(
            "You are a trading strategy agent for NSE stocks. "
            "Using the chosen stocks, market data summaries, and sentiment summaries from the conversation, "
            "for each stock:\n"
            "- Recommend Buy, Sell, or Hold\n"
            "- Suggest a near term target price in INR\n"
            "- Give a 2 to 3 line justification\n"
            "Be concise and practical for the next few trading days."
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
            "- market_data_agent looks up current market data using tools.\n"
            "- news_analyst_agent summarizes sentiment.\n"
            "- price_recommender_agent gives the final trading call.\n"
            "Call agents one by one, not in parallel. "
            "Use the earlier agents' outputs as context for later ones. "
            "At the end, respond to the user with:\n"
            "1) The 2 chosen NSE stocks (name and ticker).\n"
            "2) A short market data summary per stock.\n"
            "3) A short sentiment summary per stock.\n"
            "4) A clear Buy/Sell/Hold with target price per stock.\n"
            "Do not mention agents, tools, or internal steps."
        ),
        add_handoff_back_messages=True,
        output_mode="last_message",
    )

    app = workflow.compile()

    try:
        inputs = {
            "messages": [
                {
                    "role": "user",
                    "content": user_query,
                }
            ]
        }

        final_state = await app.ainvoke(inputs)
        messages = convert_to_messages(final_state["messages"])
        last = messages[-1]

        print("\nFinal answer:\n")
        pretty_print_message(last)

    finally:
        await cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent("Give me 2 good short term trading ideas from NSE."))
