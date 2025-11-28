# Multi-Agent Stock Recommendation System (NSE)

This project is an AI-powered multi-agent backend that generates short-term trading ideas for NSE-listed stocks using:

- LangGraph Supervisor  
- Groq LLaMA 3.3 70B  
- Bright Data MCP tools (live web search)

The system runs a structured pipeline across four agents to select stocks, gather market data, analyze news sentiment, and produce Buy/Sell/Hold recommendations.

## Features

### Multi-Agent Workflow
- **Stock Finder Agent** selects 2 promising NSE stocks.  
- **Market Data Agent** fetches price, trend, and volume using MCP tools.  
- **News Analyst Agent** summarizes recent news and sentiment.  
- **Price Recommender Agent** outputs a final trade action with target price.  
- A **supervisor** coordinates all agents and returns the final response.