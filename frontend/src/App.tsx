import { FormEvent, useMemo, useRef, useState } from "react";

type Product = {
  id: string;
  platform: string;
  title: string;
  price: number;
  currency: string;
  shipping: number;
  tax: number;
  rating: number;
  reason: string;
  url: string;
};

type Summary = {
  message: string;
  products: Product[];
  warnings: string[];
};

type AgentEvent = {
  type: string;
  thread_id: string;
  timestamp: string;
  run_id: string;
  tool?: string | null;
  message: string;
  payload: Record<string, unknown>;
};

type TaskState = {
  thread_id: string;
  status: string;
  profile?: string | null;
  provider_modes?: Record<string, string>;
  warnings?: string[];
  trace_paths?: Record<string, string>;
};

type ProviderPayload = {
  provider?: string;
  provider_mode?: string;
  latency_ms?: number;
  warnings?: string[];
};

const DEFAULT_QUERY = "我想买一套便宜又抗造的旅行三件套，预算300块，最好不要塑料的，喜欢小众一点。";

function App() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [taskState, setTaskState] = useState<TaskState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  const toolEvents = useMemo(
    () => events.filter((event) => event.tool || event.type.startsWith("subagent")),
    [events]
  );

  async function submitTask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSummary(null);
    setTaskState(null);
    setEvents([]);
    setStatus("creating");
    socketRef.current?.close();

    const response = await fetch("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });

    if (!response.ok) {
      setStatus("failed");
      setError(`任务创建失败：${response.status}`);
      return;
    }

    const data = (await response.json()) as { thread_id: string; status: string };
    setThreadId(data.thread_id);
    setStatus(data.status);
    openSocket(data.thread_id);
  }

  async function refreshTaskState(nextThreadId: string) {
    const response = await fetch(`/api/tasks/${nextThreadId}`);
    if (!response.ok) {
      return;
    }
    setTaskState((await response.json()) as TaskState);
  }

  function openSocket(nextThreadId: string) {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws/${nextThreadId}`);
    socketRef.current = socket;

    socket.onmessage = async (message) => {
      const event = JSON.parse(message.data) as AgentEvent;
      setEvents((current) => [...current, event]);
      if (event.type === "task_result") {
        const payload = event.payload as { summary?: Summary };
        if (payload.summary) {
          setSummary(payload.summary);
          setStatus("completed");
        }
        await refreshTaskState(nextThreadId);
      }
      if (event.type === "task_error") {
        setStatus("failed");
        setError(event.message);
      }
    };

    socket.onerror = () => {
      setError("WebSocket 连接异常，请确认后端服务正在运行。");
    };
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="query-panel">
          <div>
            <h1>OmniMatch</h1>
            <p>Competition shopping agent console</p>
          </div>
          <form onSubmit={submitTask}>
            <textarea value={query} onChange={(event) => setQuery(event.target.value)} />
            <button type="submit" disabled={status === "running" || status === "creating"}>
              发起任务
            </button>
          </form>
          <div className="status-row">
            <span>状态：{status}</span>
            <span>{threadId ? `线程：${threadId}` : "尚未创建任务"}</span>
            {taskState?.profile && <span>Profile：{taskState.profile}</span>}
          </div>
          {taskState?.provider_modes && (
            <div className="observability-grid">
              {Object.entries(taskState.provider_modes).map(([name, mode]) => (
                <span key={name}>
                  {name}: {mode}
                </span>
              ))}
            </div>
          )}
          {taskState?.warnings && taskState.warnings.length > 0 && (
            <div className="warning-box">
              {taskState.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          )}
          {error && <div className="error-box">{error}</div>}
        </div>

        <div className="result-panel">
          <h2>购物清单</h2>
          {summary ? (
            <>
              <p className="summary-message">{summary.message}</p>
              <div className="products">
                {summary.products.map((product) => (
                  <article className="product-card" key={product.id}>
                    <div>
                      <span className="platform">{product.platform}</span>
                      <h3>{product.title}</h3>
                    </div>
                    <p>{product.reason}</p>
                    <div className="price-row">
                      <strong>
                        {product.currency} {(product.price + product.shipping + product.tax).toFixed(2)}
                      </strong>
                      <span>评分 {product.rating.toFixed(1)}</span>
                    </div>
                  </article>
                ))}
              </div>
              {taskState?.trace_paths && (
                <div className="trace-paths">
                  {Object.entries(taskState.trace_paths).map(([name, path]) => (
                    <span key={name}>
                      {name}: {path}
                    </span>
                  ))}
                </div>
              )}
            </>
          ) : (
            <p className="empty-state">提交购物需求后，这里会展示结构化商品推荐。</p>
          )}
        </div>
      </section>

      <aside className="event-panel">
        <h2>AGUI 事件流</h2>
        <div className="trace-stats">
          <span>{events.length} 个事件</span>
          <span>{toolEvents.length} 个工具/子 Agent 事件</span>
        </div>
        <div className="event-list">
          {events.map((event, index) => (
            <div className="event-item" key={`${event.run_id}-${index}`}>
              <div className="event-meta">
                <span>{event.type}</span>
                {event.tool && <span>{event.tool}</span>}
              </div>
              <ProviderMeta event={event} />
              <p>{event.message}</p>
            </div>
          ))}
          {events.length === 0 && <p className="empty-state">事件会实时显示在这里。</p>}
        </div>
      </aside>
    </main>
  );
}

function ProviderMeta({ event }: { event: AgentEvent }) {
  if (!event.type.startsWith("provider_")) {
    return null;
  }
  const payload = event.payload as ProviderPayload;
  return (
    <div className="provider-meta">
      {payload.provider && <span>{payload.provider}</span>}
      {payload.provider_mode && <span>{payload.provider_mode}</span>}
      {typeof payload.latency_ms === "number" && <span>{payload.latency_ms} ms</span>}
      {payload.warnings?.map((warning) => <span key={warning}>{warning}</span>)}
    </div>
  );
}

export default App;
