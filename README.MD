# Multi Agent Stock Recommender

## Overview
This project demonstrates a multi agent stock analysis workflow using LangGraph, Groq, and Bright Data MCP. A React frontend sends a query and intent, and the backend returns two stock recommendations plus a stepwise timeline visualization.

## Architecture

### Backend
- **FastAPI** handles API requests.
- **LangGraph supervisor** runs a fixed 4 agent sequence:
  1. `stock_finder_agent` selects two NASDAQ/NYSE stocks.
  2. `market_data_agent` fetches recent prices, volume, and trend.
  3. `news_analyst_agent` summarizes short term sentiment.
  4. `price_recommender_agent` outputs final structured JSON.
- **Groq Llama 3.1 8B** model powers all agent reasoning.
- **Bright Data MCP** provides data and scraping tools.
- API response includes:
  - `stocks`: final JSON output for the UI
  - `timeline`: every message from supervisor and agents

### Frontend
- **React + TypeScript**
- User inputs:
  - Text query
  - Intent toggle (buy or sell)
- UI displays:
  - Two stock cards with current and target price, trend, sentiment, and reason
  - A pure CSS vertical timeline describing each internal pipeline step



### Environment Variables
- BRIGHT_DATA_API_KEY=
- WEB_UNLOCKER_ZONE=unblocker
- BROWSER_ZONE=scraping_browser
- GROQ_API_KEY=yourkey
- GROQ_MODEL=model


### Known Hiccups

-  Graph recursion: If the supervisor prompt is changed or relaxed, agents may loop. Keep the strict ordering.

-  High token use: Full graph with 4 agents + MCP calls is expensive. Groq helps but still uses multiple completions per request.

-  Use graph_app_dev graph for dev: It uses a single model call and returns both stocks and timeline. This keeps token usage low.

### Dev Mode

- Switch the backend imports to use the dev graph:

from graph_app_dev import build_graph, extract_final_text, build_timeline_with_result
