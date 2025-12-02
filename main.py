import asyncio
import json

from dotenv import load_dotenv
from graph_app_dev import (
    build_graph,
    extract_final_text,
    build_timeline_with_result,
)

load_dotenv()


async def run_agent(user_query: str):
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
        print("\nRAW graph_output['messages']:\n")
        try:
            print(json.dumps(final_state["messages"], indent=2))
        except Exception:
            print(final_state["messages"])
      
        answer = extract_final_text(final_state)

        print("\nFinal answer:\n")
        print(json.dumps(answer, indent=2))

        # Build timeline 
        timeline_bundle = build_timeline_with_result(final_state)

        print("\nTimeline object:\n")
        print(json.dumps(timeline_bundle, indent=2))

    finally:
        await cleanup()


if __name__ == "__main__":
    query = "User intent: BUY. User query: Give me 2 good short term trading ideas."
    asyncio.run(run_agent(query))
