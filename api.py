import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph_app import build_graph, extract_final_text


graph = None
cleanup_cb = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, cleanup_cb

    # Startup
    graph, cleanup_cb = await build_graph()

    try:
        yield
    finally:
        # Shutdown
        if cleanup_cb:
            await cleanup_cb()


app = FastAPI(lifespan=lifespan)


# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # relax for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str


class RecommendResponse(BaseModel):
    answer: str


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    if graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    inputs = {
        "messages": [
            {
                "role": "user",
                "content": req.query,
            }
        ]
    }

    result = await graph.ainvoke(inputs)
    answer_text = extract_final_text(result)

    return RecommendResponse(answer=answer_text)
