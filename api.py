import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph_app import build_graph, extract_final_text, build_timeline_with_result

logging.basicConfig(
    level=logging.INFO,
    format="API LOG | %(message)s",
)
log = logging.getLogger("api")

graph = None
cleanup_cb = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, cleanup_cb

    log.info("Starting graph...")
    graph, cleanup_cb = await build_graph()
    log.info("Graph initialized")

    try:
        yield
    finally:
        log.info("Shutting down graph...")
        if cleanup_cb:
            await cleanup_cb()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str
    intent: str  # buy or sell


class Stock(BaseModel):
    name: str
    ticker: str
    action: str
    targetPrice: str
    currentPrice: str
    trend: str
    sentiment: str
    reason: str


class TimelineEvent(BaseModel):
    step: int
    role: str
    agent: str
    content: str
    label: Optional[str] = None


class RecommendResponse(BaseModel):
    stocks: List[Stock]
    timeline: List[TimelineEvent]


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    if graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    intent_upper = req.intent.strip().upper()
    if intent_upper not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="intent must be 'buy' or 'sell'")

    log.info(f"Incoming request: query='{req.query}', intent='{req.intent}'")

    inputs = {
        "messages": [
            {
                "role": "user",
                "content": f"User intent: {intent_upper}. User query: {req.query}",
            }
        ]
    }

    log.info("Running graph...")

    graph_output = await graph.ainvoke(
        inputs,
        config={"recursion_limit": 6},
    )

    try:
        pretty = json.dumps(graph_output, indent=2)
        log.info(f"Raw graph output:\n{pretty}")
    except Exception:
        log.info("Graph output (non-JSON):")
        log.info(str(graph_output))

    bundle = build_timeline_with_result(graph_output)
    data = bundle.get("final", {}) or {}
    timeline_raw = bundle.get("timeline", []) or []

    log.info(f"Parsed structured data: {data}")
    log.info(f"Timeline events count: {len(timeline_raw)}")

    if "stocks" not in data or not isinstance(data["stocks"], list):
        log.info("No valid stocks found in model output. Returning empty list.")
        data = {"stocks": []}

    try:
        timeline: List[TimelineEvent] = [TimelineEvent(**evt) for evt in timeline_raw]
    except Exception as e:
        log.info(f"Failed to parse timeline events: {e}")
        timeline = []

    log.info("Sending response to frontend")

    return RecommendResponse(stocks=data["stocks"], timeline=timeline)
