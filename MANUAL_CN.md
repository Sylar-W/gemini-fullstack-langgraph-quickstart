# 项目概述

本项目旨在开发一个全栈研究代理应用。该应用利用 LangGraph 框架和 Gemini 模型，为用户提供强大而智能的研究能力。

其核心功能在于接收用户提出的问题，并通过一系列先进的自动化流程来寻找和生成答案。这些流程包括：

*   **动态网络搜索**：根据用户问题，在互联网上进行广泛而有针对性的信息搜集。
*   **反思与迭代优化**：对搜索结果进行评估和筛选，通过多轮反思和迭代，不断优化答案的准确性和相关性。
*   **引用与溯源**：最终生成的答案将包含清晰的引用来源，方便用户追溯和核实信息。

通过以上功能，本项目致力于为用户提供一个高效、可靠、可信的研究助手。

# 技术栈

本项目采用前后端分离的架构，具体技术选型如下：

## 前端

*   **React**: 用于构建用户界面的 JavaScript 库。
*   **Vite**: 下一代前端构建工具，提供极速的冷启动和模块热更新。
*   **Tailwind CSS**: 一个功能类优先的 CSS 框架，用于快速构建自定义设计。
*   **Shadcn UI**: 一套可重用的 UI 组件，基于 Radix UI 和 Tailwind CSS 构建。
*   **@langchain/langgraph-sdk/react**: LangGraph SDK 的 React 版本，方便前端与 LangGraph 后端集成。

## 后端

*   **Python**: 主要的后端编程语言。
*   **FastAPI**: 一个现代、快速（高性能）的 Python web 框架，用于构建 API。
*   **LangGraph**: 用于构建可组合、可流式处理的 AI 应用的框架。
*   **Google Gemini API**: Google 提供的大型语言模型服务，用于驱动应用的核心智能。

# 项目架构

## 前端 (Frontend)

前端负责提供用户交互界面，允许用户输入问题，并实时展示聊天记录以及研究代理的活动状态。

*   **职责**:
    *   构建用户友好的交互界面。
    *   接收并提交用户输入的研究问题。
    *   展示与代理的聊天记录。
    *   实时显示代理的内部活动和思考过程。

*   **主要组件**:
    *   `App.tsx`: 作为应用的主入口和核心逻辑容器，负责整体布局、状态管理（如使用 `useState` 和 `useReducer`），以及通过 SDK 与后端 LangGraph Agent 进行通信。
    *   `ChatMessagesView.tsx`: 用于渲染聊天气泡，清晰展示用户提问和代理的回答。
    *   `ActivityTimeline.tsx`: 以时间线的形式，可视化展示 LangGraph Agent 在执行任务过程中的每一步活动和中间结果，增强透明度。
    *   `WelcomeScreen.tsx`: 应用的初始欢迎界面，引导用户开始新的研究会话。
    *   `InputForm.tsx`: 集成在 `WelcomeScreen` 和 `ChatMessagesView` 中，提供用户输入问题的表单控件。

*   **与后端通信**:
    *   前端使用 `@langchain/langgraph-sdk/react` 包提供的 `useStream` Hook。
    *   通过 HTTP 和 Server-Sent Events (SSE) 实现与后端 LangGraph Agent 的流式通信，确保代理活动的实时更新和最终答案的顺畅传递。

## 后端 (Backend)

后端是整个应用的核心，负责实现研究代理的复杂逻辑，处理前端发来的用户请求，执行信息检索、内容分析、知识反思，并最终生成结构化的答案。

*   **职责**:
    *   实现 LangGraph Agent 的核心研究和推理逻辑。
    *   处理来自前端的用户请求。
    *   执行动态网络搜索和信息提取。
    *   对信息进行分析、评估和迭代优化。
    *   生成包含引用的最终答案。

*   **API 服务**:
    *   使用 **FastAPI** (`backend/src/agent/app.py`) 构建 RESTful API 服务。
    *   该服务不仅处理代理相关的请求，还负责提供前端静态文件的托管。

*   **LangGraph Agent (`backend/src/agent/graph.py`)**:
    这是研究代理的核心，通过 LangGraph 构建为一个状态图 (State Graph)，定义了任务执行的流程和逻辑。
    *   **状态定义 (`backend/src/agent/state.py`)**:
        *   `OverallState`: TypedDict，定义了整个 Agent 图的全局状态，包含用户问题、搜索查询、搜索结果、反思、最终答案等。
        *   `ReflectionState`: TypedDict，用于 `reflection` 节点的子图状态，包含当前搜索结果和已进行的反思。
    *   **工具与模式 (`backend/src/agent/tools_and_schemas.py`)**:
        *   定义了 Agent 在与 LLM (Large Language Model) 交互时所使用的 Pydantic 数据结构（模式），确保数据的一致性和校验。
        *   例如 `SearchQueryList` 用于结构化初始生成的搜索查询列表，`Reflection` 用于结构化反思过程的输出。
        *   还可能包含 LangChain Tools，如用于执行网络搜索的工具。
    *   **核心流程节点 (Nodes)**: 图中的每个节点代表一个处理步骤，通常是一个 Python 函数或可调用对象。
        *   `generate_query`: 接收用户问题，利用 LLM 生成一组初始的、多样化的搜索查询。
        *   `web_research`: 根据生成的查询，并行执行网络搜索，收集相关信息。
        *   `reflection`: 对 `web_research` 的结果进行分析和评估，LLM 在此步骤中识别当前信息的不足之处（知识差距）并提出改进建议（例如，生成新的搜索查询）。
        *   `evaluate_research` (Conditional Edge): 这是一个条件节点，根据 `reflection` 的结果（例如，是否还有未解决的知识差距，或是否已达到最大迭代次数）来决定下一步是继续进行新一轮的 `web_research` 还是进入 `finalize_answer` 阶段。
        *   `finalize_answer`: 当 `evaluate_research` 决定结束研究过程时，此节点负责综合所有收集到的信息和反思，利用 LLM 生成最终的、带引用的答案。
    *   **配置 (`backend/src/agent/configuration.py`)**:
        *   定义了 Agent 的可配置参数，方便调整和优化。
        *   例如：选择使用的 Gemini 模型版本 (e.g., `gemini-1.5-pro-latest`)，初始并行搜索查询的数量，研究过程的最大迭代（循环）次数等。
    *   **提示 (`backend/src/agent/prompts.py`)**:
        *   存储了所有用于指导 LLM 在不同节点（如 `generate_query`, `reflection`, `finalize_answer`）生成特定内容的提示模板 (Prompt Templates)。这些模板精心设计，以确保 LLM 的输出符合预期。

# 核心实现逻辑详解

LangGraph Agent 的核心逻辑围绕着一个迭代的研究循环，该循环旨在通过动态搜索和反思来逐步优化答案的质量。

1.  **启动与配置**:
    *   用户在前端界面提交研究问题。
    *   用户可以选择不同的“努力程度”（Effort）和“推理模型”（Reasoning Model）。这些选项会影响后端 Agent 的行为，例如，“努力程度”可能映射到最大研究循环次数或初始查询数量，“推理模型”则指定了用于生成、反思和总结的 Gemini 模型版本。
    *   这些参数连同用户问题一起被发送到后端 FastAPI 应用，并作为 LangGraph Agent 的初始输入 `OverallState`。

2.  **查询生成 (`generate_query`节点)**:
    *   此节点接收 `OverallState` 中的用户原始问题和配置参数（如 `initial_query_count`，来自 `configuration.py`）。
    *   它使用指定的 Gemini 模型（通过 `genai_client` 访问，模型名称来自配置）和 `prompts.py` 中的 `query_writer_instructions` 提示模板。
    *   LLM 的任务是根据用户问题，生成一个结构化的 `SearchQueryList` (定义于 `tools_and_schemas.py`)，其中包含多个（数量由 `initial_query_count` 决定）不同的搜索查询字符串。这些查询旨在从不同角度探索用户的问题。
    *   生成的 `SearchQueryList` 更新到 `OverallState` 的 `search_queries` 字段。

3.  **并行网页搜索 (`web_research`节点)**:
    *   此节点接收 `OverallState` 中的 `search_queries`。
    *   对于 `search_queries` 列表中的每一个查询字符串，它会并行启动一个搜索任务。
    *   每个搜索任务利用 Gemini 模型（通常是支持工具使用的版本）和 Google Search API 工具（通过 `genai_client.tools(["google_search"])`）。
    *   LLM 会根据查询调用 Google Search 工具获取原始搜索结果。
    *   节点内部逻辑会进一步处理这些原始结果：
        *   提取关键信息和网页片段。
        *   记录每个结果的引用来源（URL）。
        *   尝试将可能的短链接（如 Google 搜索结果中的 t.co 链接）解析为其原始的、完整的 URL，以提高引用的可靠性和持久性。
    *   所有并行搜索任务收集到的信息（文本片段和对应的引用）被汇总并更新到 `OverallState` 的 `search_results` 字段。同时，`research_loop_count` 会递增。

4.  **反思与知识差距分析 (`reflection`节点)**:
    *   此节点接收 `OverallState` 中的 `search_results`（当前收集到的所有信息）和原始用户问题。
    *   它使用指定的 Gemini 模型和 `prompts.py` 中的 `reflection_instructions` 提示模板。
    *   LLM 的任务是全面评估当前 `search_results` 是否足以回答用户的原始问题。它需要识别：
        *   信息是否已经充分 (`is_sufficient`: boolean)。
        *   如果信息不充分，还存在哪些具体的知识空白点或未解答的方面 (`knowledge_gap`: string)。
        *   针对这些知识空白点，建议生成哪些新的、更具针对性的后续搜索查询 (`follow_up_queries`: list of strings)。
    *   这些分析结果被结构化为一个 `Reflection` 对象 (定义于 `tools_and_schemas.py`) 并更新到 `OverallState` 的 `reflection_history` 字段（通常是一个列表，记录每次反思的结果）。

5.  **评估与迭代 (`evaluate_research`条件边)**:
    *   这是一个 LangGraph 中的条件边 (conditional edge)，它决定了研究流程的走向。
    *   它检查 `reflection` 节点最新输出的 `Reflection` 对象中的 `is_sufficient` 标志，以及 `OverallState` 中的当前研究循环次数 `research_loop_count`。
    *   **迭代**: 如果 `is_sufficient` 为 `False` **并且** `research_loop_count` 小于配置中定义的 `max_research_loops` (`configuration.py`)，则条件边会将流程导向 `web_research` 节点。此时，`reflection` 节点生成的 `follow_up_queries` 会被用作下一轮 `web_research` 的输入。
    *   **终止**: 如果 `is_sufficient` 为 `True` **或者** `research_loop_count` 已达到 `max_research_loops`，则条件边会将流程导向 `finalize_answer` 节点。

6.  **最终答案生成 (`finalize_answer`节点)**:
    *   此节点在研究循环结束后被调用。
    *   它接收包含所有累积的 `search_results`、`reflection_history` 和原始用户问题的 `OverallState`。
    *   它使用指定的 Gemini 模型和 `prompts.py` 中的 `answer_instructions` 提示模板。
    *   LLM 的任务是综合所有可用的信息，生成一个全面、连贯、准确回答用户原始问题的最终答案。
    *   答案应包含清晰的引用，指向其信息来源的 URL（来自 `search_results`）。
    *   生成的最终答案作为一个 AI 类型的消息 (AIMessage) 被添加到 `OverallState` 的 `messages` 列表中，准备好展示给用户。

7.  **流式输出 (Streaming via SSE)**:
    *   在整个 LangGraph Agent 的执行过程中，每当 `OverallState`发生变化（例如，在每个节点执行完毕后），这些变化会通过 FastAPI 的 Server-Sent Events (SSE) 连接流式传输到前端。
    *   前端 `App.tsx` 中的 `useStream` Hook 监听这些事件。其 `onUpdateEvent` 回调函数负责接收这些状态更新。
    *   `onUpdateEvent` 会解析收到的数据，提取出最新的代理活动、中间结果（如生成的查询、搜索结果片段、反思内容）以及最终答案。
    *   这些信息被用来实时更新 `ActivityTimeline` 和 `ChatMessagesView` 等UI组件，让用户能够看到研究过程的进展和最终结果。

这个迭代和反思的循环是该研究代理能够处理复杂问题并提供高质量、带引用答案的关键。通过动态调整搜索策略和评估信息充分性，代理能够更有效地逼近用户的真实需求。

# 项目运行与部署

本节概述了如何在本地运行项目进行开发测试，以及如何使用 Docker 进行构建和部署。详细信息请参考项目根目录下的 `README.md` 文件。

## 本地开发环境

### 1. 先决条件 (Prerequisites)

*   **Node.js**: 用于运行前端 Vite 开发服务器和构建前端资源。建议使用最新 LTS 版本。
*   **Python**: 用于运行后端 FastAPI 应用和 LangGraph Agent。建议使用 Python 3.9 或更高版本。
*   **Google API 密钥**: 需要配置有效的 Google API 密钥，以授权 Gemini API 和 Google Search API 的使用。通常通过环境变量 `GOOGLE_API_KEY` 进行配置。
*   **Poetry**: (推荐) 用于管理 Python 依赖。如果使用 Poetry，请确保已安装。

### 2. 安装依赖 (Install Dependencies)

*   **前端**:
    进入 `frontend` 目录，运行以下命令安装 Node.js 依赖：
    ```bash
    cd frontend
    npm install
    # 或者使用 yarn:
    # yarn install
    cd ..
    ```

*   **后端**:
    进入 `backend` 目录。如果使用 Poetry，运行：
    ```bash
    cd backend
    poetry install
    cd ..
    ```
    如果未使用 Poetry，但有 `requirements.txt` 文件，可以创建一个虚拟环境并使用 pip 安装：
    ```bash
    cd backend
    python -m venv .venv
    source .venv/bin/activate # 或者在 Windows 上: .venv\Scripts\activate
    pip install -r requirements.txt
    cd ..
    ```
    (请确保 `requirements.txt` 文件存在且包含所有必要的依赖，如 `fastapi`, `uvicorn`, `langgraph`, `google-generativeai`, `langchain-google-genai` 等。)

### 3. 运行开发服务器 (Run Development Servers)

*   **后端 FastAPI 服务**:
    进入 `backend` 目录。如果使用 Poetry，运行：
    ```bash
    cd backend
    poetry run uvicorn src.agent.app:app --reload --port 8000
    ```
    如果未使用 Poetry (并已激活虚拟环境)，运行：
    ```bash
    cd backend
    uvicorn src.agent.app:app --reload --port 8000
    ```
    后端服务通常运行在 `http://localhost:8000`。

*   **前端 Vite 开发服务**:
    打开一个新的终端窗口，进入 `frontend` 目录，运行：
    ```bash
    cd frontend
    npm run dev
    ```
    前端开发服务器通常运行在 `http://localhost:5173`，并会自动代理 API 请求到后端服务（通常在 `vite.config.ts` 中配置）。

配置完成后，在浏览器中打开前端服务的 URL (例如 `http://localhost:5173`) 即可与应用交互。

## Docker 构建与部署

项目支持使用 Docker 进行容器化构建和部署，简化了环境配置和依赖管理。

### 1. 构建 Docker 镜像

在项目根目录下，通常会有一个 `Dockerfile` (或者前端和后端分别有各自的 `Dockerfile` 及一个 `docker-compose.yml` 用于编排)。

*   **单个 Dockerfile (合并前端构建和后端服务)**:
    如果项目使用单个 `Dockerfile` 来构建包含前端静态资源和后端服务的镜像，可以运行：
    ```bash
    docker build -t research-agent-app .
    ```

*   **使用 Docker Compose (推荐)**:
    如果项目包含 `docker-compose.yml` 文件，它会定义前端和后端服务，并处理它们之间的网络。
    在项目根目录下运行：
    ```bash
    docker-compose build
    ```

### 2. 运行 Docker 容器

*   **单个 Docker 镜像**:
    运行之前构建的镜像：
    ```bash
    docker run -p 80:8000 -e GOOGLE_API_KEY="YOUR_API_KEY" research-agent-app
    ```
    (将 `"YOUR_API_KEY"`替换为你的实际 Google API 密钥。端口映射 `-p 80:8000` 表示将主机的 80 端口映射到容器的 8000 端口。)

*   **使用 Docker Compose**:
    在项目根目录下运行：
    ```bash
    docker-compose up
    ```
    (确保在 `docker-compose.yml` 或关联的 `.env` 文件中配置了必要的环境变量如 `GOOGLE_API_KEY`。)

使用 Docker Compose 是推荐的方式，因为它能更好地管理多服务应用的构建、网络和运行。请查阅项目中的 `Dockerfile` 和 `docker-compose.yml` (如果存在) 以获取确切的构建和运行指令。
