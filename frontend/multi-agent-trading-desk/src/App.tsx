import "./App.css";
import { useState } from "react";

type Intent = "buy" | "sell";

type Stock = {
  name: string;
  ticker: string;
  action: string;
  targetPrice: string;
  currentPrice: string;
  trend: string;
  sentiment: string;
  reason: string;
};

type TimelineEvent = {
  step: number;
  role: string;
  agent: string;
  content: string;
  label?: string;
};

type ResultState = {
  stocks: Stock[];
  timeline?: TimelineEvent[];
};

function App() {
  const [query, setQuery] = useState("");
  const [intent, setIntent] = useState<Intent>("buy");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResultState>({
    stocks: [],
    timeline: [],
  });

  async function runQuery() {
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, intent }),
      });

      if (!res.ok) throw new Error("Backend error");

      const data = await res.json();
      setResult(data as ResultState);
    } catch (e) {
      console.error(e);
      setResult({ stocks: [], timeline: [] });
    }

    setLoading(false);
  }

  const isBuy = intent === "buy";

  return (
    <div className="app-shell">
      <main className="main">
        <div className="panel">
          <div>
            <div className="topbar">
              <div className="topbar-title-block">
                <span className="topbar-title">
                  Multi Agent Stock Recommender
                </span>
              </div>
            </div>
            <div className="page">
              <div className="query-panel">
                <div className="query-panel-title-block">
                  <span className="query-panel-title">Ask for stock picks</span>
                  <span className="query-panel-subtitle">
                    The multi agent system will process your query
                  </span>
                </div>

                <div className="query-panel-row">
                  <input
                    className="input-text"
                    placeholder="Example. 2 short term NSE stocks"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  {/* Intent toggle */}
                  <div className="query-panel-row">
                    <div className="intent-toggle-wrapper">
                      {/* <span className="intent-label">Intent</span> */}
                      <label className="intent-toggle">
                        <input
                          type="checkbox"
                          checked={isBuy}
                          onChange={() =>
                            setIntent((prev) =>
                              prev === "buy" ? "sell" : "buy"
                            )
                          }
                        />
                        <span className="intent-toggle-track">
                          <span
                            className={`intent-toggle-text ${
                              isBuy ? "active" : ""
                            }`}
                          >
                            Buy
                          </span>
                          <span
                            className={`intent-toggle-text ${
                              !isBuy ? "active" : ""
                            }`}
                          >
                            Sell
                          </span>
                        </span>
                        <span
                          className={`intent-toggle-thumb ${
                            isBuy ? "left" : "right"
                          }`}
                        />
                      </label>
                    </div>
                  </div>
                  <button
                    className="btn-primary"
                    onClick={runQuery}
                    disabled={loading}
                  >
                    {loading ? "Thinking…" : "Generate"}
                  </button>
                </div>
              </div>

              <div >
                <div className="stock-cards">
                  {result &&
                    result.stocks &&
                    result.stocks.map((stock, index) => (
                      <div className="stock-card" key={index}>
                        <div className="stock-card-header">
                          <div className="stock-card-title-block">
                            <span className="stock-card-name">
                              {stock.name}
                            </span>
                            <span className="stock-card-ticker">
                              {stock.ticker}
                            </span>
                          </div>
                          <span
                            className={`stock-card-action stock-card-action--${stock.action.toLowerCase()}`}
                          >
                            {stock.action}
                          </span>
                        </div>

                        <div className="stock-card-body">
                          <div className="stock-card-row">
                            <div>
                              <span className="label">Current Price: </span>
                              <span className="value">
                                {stock.currentPrice}
                              </span>
                            </div>
                            <div>
                              <span className="label">Target: </span>
                              <span className="value">{stock.targetPrice}</span>
                            </div>
                          </div>

                          <div className="stock-card-row">
                            <div>
                              <span className="label">Trend: </span>
                              <span className="value">{stock.trend}</span>
                            </div>
                            <div>
                              <span className="label">Sentiment: </span>
                              <span className="value">{stock.sentiment}</span>
                            </div>
                          </div>

                          <div className="stock-card-reason">
                            {stock.reason}
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
          <div className="timeline-block">
            <h2 className="timeline-title">Execution Timeline</h2>

            {result.timeline && result.timeline.length > 0 ? (
              <div className="timeline-container">
                <div className="timeline-line"></div>

                {result.timeline.map((e, idx) => (
                  <div className="timeline-item" key={idx}>
                    <div className="timeline-marker"></div>

                    <div className="timeline-content">
                      <div className="timeline-step">Step {e.step}</div>
                      <div className="timeline-agent">{e.agent}</div>
                      <div className="timeline-text">{e.content}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p>No timeline yet.</p>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
