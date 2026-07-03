# OmniMatch

OmniMatch 是一个可运行的对话式跨平台购物 Agent mock MVP。

当前版本重点实现 `idea.md` 中的教学骨架：

- FastAPI 任务 API
- WebSocket AGUI 风格事件流
- mock `Think -> Act -> Observe -> Reflect` AgentLoop
- 跨平台搜索场景下的 mock 同质子 Agent fork
- React/Vite 单页教学演示台
- 可导入的 recall、memory、compression、eval、prompt、utils stub 模块

当前 MVP 不调用真实 LLM、电商 API、向量库、Redis 或数据库。

## 后端启动

在项目根目录安装后端依赖：

```bash
uv sync
```

启动 FastAPI 后端服务：

```bash
uv run uvicorn app.api.server:app --reload
```

后端默认监听：

```text
http://127.0.0.1:8000
```

运行后端测试：

```bash
uv run pytest -v
```

## 前端启动

进入前端目录并安装依赖：

```bash
cd frontend
npm install
```

启动 Vite 前端开发服务：

```bash
cd frontend
npm run dev
```

前端默认监听：

```text
http://127.0.0.1:5173
```

前端会把 `/api` 和 `/ws` 请求代理到后端 `http://127.0.0.1:8000`。

构建前端：

```bash
cd frontend
npm run build
```

## 本地运行顺序

1. 打开第一个终端，在项目根目录启动后端：

```bash
uv run uvicorn app.api.server:app --reload
```

2. 打开第二个终端，启动前端：

```bash
cd frontend
npm run dev
```

3. 浏览器访问：

```text
http://127.0.0.1:5173
```

4. 在页面中提交默认购物需求。
5. 右侧会实时显示 AGUI 事件流。
6. 左侧会展示最终 mock 商品清单和商品卡片。

最终 summary 会写入：

```text
output/{thread_id}/summary.json
```

## CLI 示例

不启动 Web 服务，直接运行 mock AgentLoop：

```bash
uv run python examples/run_mock_agent.py
```
