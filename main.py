import asyncio

from dotenv import load_dotenv
from graph_app import build_graph, extract_final_text

load_dotenv()


async def run_agent(user_query: str):
    # Build the multi agent graph and get cleanup callback
    app, cleanup = await build_graph()

    try:
        inputs = {
            "messages": [
                {
                    "role": "user",
                    "content": user_query,
                }
            ]
        }

        # Run the graph
        final_state = await app.ainvoke(inputs)

        # Extract final text answer
        answer = extract_final_text(final_state)

        print("\nFinal answer:\n")
        print(answer)

    finally:
        # Make sure MCP connections are cleaned up
        await cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent("Give me 2 good short term trading ideas from NSE."))
