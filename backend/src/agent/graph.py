import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
# google.genai Client is removed as it's no longer directly used here for search
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate # For potential agent prompt
from langchain.agents import create_tool_calling_agent, AgentExecutor # For Ollama/Azure agent

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

# GEMINI_API_KEY check remains relevant for Gemini provider
if os.getenv("GEMINI_API_KEY") is None and os.getenv("MODEL_PROVIDER", "gemini").lower() == "gemini":
    raise ValueError("GEMINI_API_KEY is not set for Gemini provider.")
# We no longer initialize genai_client globally here for search.
# Search tool will use its own configured API key.


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search query for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init LLM based on provider
    if configurable.model_provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=configurable.query_generator_model,
            temperature=1.0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    elif configurable.model_provider == "ollama":
        if not configurable.ollama_api_base_url or not configurable.ollama_model_name:
            raise ValueError(
                "OLLAMA_API_BASE_URL and OLLAMA_MODEL_NAME must be set for Ollama provider"
            )
        llm = ChatOpenAI(
            model_name=configurable.ollama_model_name,
            openai_api_base=configurable.ollama_api_base_url,
            openai_api_key="NA",  # Typically not required for local Ollama
            temperature=1.0,
            # max_retries might not be a direct param for ChatOpenAI, check docs if needed
        )
    elif configurable.model_provider == "azure_openai":
        if not configurable.azure_openai_api_version or \
           not configurable.azure_openai_deployment_name or \
           not configurable.azure_openai_api_base_url or \
           not configurable.azure_openai_api_key:
            raise ValueError(
                "Azure OpenAI configurations (API_VERSION, DEPLOYMENT_NAME, API_BASE_URL, API_KEY) must be set"
            )
        llm = AzureChatOpenAI(
            openai_api_version=configurable.azure_openai_api_version,
            azure_deployment=configurable.azure_openai_deployment_name,
            azure_endpoint=configurable.azure_openai_api_base_url,
            api_key=configurable.azure_openai_api_key,
            temperature=1.0,
            # max_retries might not be a direct param for AzureChatOpenAI, check docs if needed
        )
    else:
        raise ValueError(f"Unsupported model provider: {configurable.model_provider}")

    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    configurable = Configuration.from_runnable_config(config)

    if not configurable.google_api_key or not configurable.google_cse_id:
        # Skip web research if API keys are not configured
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [
                "Web research skipped due to missing Google API Key or CSE ID configuration."
            ],
        }

    search_tool = Tool(
        name="google_search",
        description="Performs a Google search and returns results.",
        func=GoogleSearchAPIWrapper(
            google_api_key=configurable.google_api_key,
            google_cse_id=configurable.google_cse_id,
        ).run,
    )

    current_date = get_current_date()
    # The web_searcher_instructions might need to be more generic or conditional
    # For now, we use it as a base for the input to the LLM/agent.
    formatted_prompt_for_llm = web_searcher_instructions.format(
        current_date=current_date,
        research_topic=state["search_query"],
    )

    modified_text = ""
    sources_gathered = []

    try:
        if configurable.model_provider == "gemini":
            llm = ChatGoogleGenerativeAI(
                model=configurable.query_generator_model, # Using query_generator_model for search tasks
                temperature=0,
                max_retries=2,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
            # Bind the tool to the LLM. Gemini should handle tool calling and response formatting.
            llm_with_tools = llm.bind_tools([search_tool], tool_choice="google_search")

            ai_msg = llm_with_tools.invoke(formatted_prompt_for_llm)

            # Attempt to process Gemini output - this is the most fragile part
            # LangChain's `bind_tools` with Gemini might not return grounding_metadata
            # in the same way as the direct genai_client did. This needs testing.
            # For now, assume ai_msg.content is the main text and try to parse tool_calls if any.
            modified_text = ai_msg.content

            # Simplified source gathering for Gemini with bound tool:
            # If Gemini's bound tool provides structured output or if tool_calls are inspectable
            if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    if tool_call['name'] == 'google_search':
                        # The output from wrapper.run is usually a string of search results.
                        # We need to parse this string to find URLs and snippets.
                        # This is a placeholder for more robust parsing.
                        search_results_str = tool_call.get('output', "") # Actual output depends on how LLM formats it
                        # Basic URL extraction from the search result string
                        import re
                        urls = re.findall(r'https?://\S+', search_results_str)
                        for i, url in enumerate(urls):
                             # Create a simplified source entry
                            source_entry = {
                                "value": url,
                                "short_url": f"[{state['id']}-{i+1}]", # Create a unique enough marker
                                "id": f"{state['id']}-{i+1}",
                                "title": f"Search Result {i+1}", # Placeholder title
                                "segments": [url] # Simplified segment
                            }
                            sources_gathered.append(source_entry)
                            modified_text = modified_text.replace(url, source_entry["short_url"])


            elif hasattr(ai_msg, 'additional_kwargs') and ai_msg.additional_kwargs.get('tool_calls'):
                # Anthropic and some other models might use this format
                tool_calls = ai_msg.additional_kwargs.get('tool_calls', [])
                for call in tool_calls:
                    if call.function.name == 'google_search':
                        # This part would need to be adapted based on actual tool call output structure
                        # For now, this is a conceptual placeholder
                        pass # Add logic to parse call.function.arguments and tool output if available

            # Fallback if no clear tool calls with output are found, but we have content
            if not sources_gathered and modified_text:
                # Basic URL extraction from the LLM's content if no explicit tool output found
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', modified_text)
                for i, url in enumerate(list(set(urls))): # Unique URLs
                    source_id = f"{state['id']}-{i+1}"
                    short_url_marker = f"[{source_id}]"
                    sources_gathered.append({
                        "value": url,
                        "short_url": short_url_marker,
                        "id": source_id,
                        "title": url, # Simplified title
                        "segments": [url]
                    })
                    modified_text = modified_text.replace(url, short_url_marker)


        elif configurable.model_provider in ["ollama", "azure_openai"]:
            if configurable.model_provider == "ollama":
                if not configurable.ollama_api_base_url or not configurable.ollama_model_name:
                    raise ValueError("Ollama config missing")
                llm = ChatOpenAI(
                    model_name=configurable.ollama_model_name,
                    openai_api_base=configurable.ollama_api_base_url,
                    openai_api_key="NA", temperature=0
                )
            else: # azure_openai
                if not configurable.azure_openai_api_version or \
                   not configurable.azure_openai_deployment_name or \
                   not configurable.azure_openai_api_base_url or \
                   not configurable.azure_openai_api_key:
                    raise ValueError("Azure OpenAI config missing")
                llm = AzureChatOpenAI(
                    openai_api_version=configurable.azure_openai_api_version,
                    azure_deployment=configurable.azure_openai_deployment_name,
                    azure_endpoint=configurable.azure_openai_api_base_url,
                    api_key=configurable.azure_openai_api_key, temperature=0
                )

            # Create a simple agent to use the search tool
            # The prompt for this agent might need to be more specific to encourage summarization of search results
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant that uses Google Search to answer questions. Summarize the findings."),
                ("user", "{input}")
            ])
            agent = create_tool_calling_agent(llm, [search_tool], agent_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

            result = agent_executor.invoke({"input": formatted_prompt_for_llm})
            modified_text = result.get("output", "")

            # Simplified source gathering for Ollama/Azure:
            # Extract URLs from the agent's final output text
            import re
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', modified_text)
            for i, url in enumerate(list(set(urls))): # Unique URLs
                source_id = f"{state['id']}-{i+1}"
                short_url_marker = f"[{source_id}]"
                sources_gathered.append({
                    "value": url,
                    "short_url": short_url_marker,
                    "id": source_id,
                    "title": url, # Simplified title
                    "segments": [url]
                })
                # Replace URL with marker in the text. Do this carefully to avoid breaking markdown or other syntax.
                # This replacement is basic and might need improvement.
                modified_text = modified_text.replace(url, short_url_marker)

        else:
            raise ValueError(f"Unsupported model provider for web_research: {configurable.model_provider}")

    except Exception as e:
        print(f"Error during web research with {configurable.model_provider}: {e}")
        modified_text = f"Error performing web research: {e}"
        # sources_gathered remains empty

    # Ensure web_research_result is always a list of strings
    final_web_research_result = [modified_text if modified_text else "No information gathered from web research."]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]], # Keep track of the query used for this result
        "web_research_result": final_web_research_result,
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model based on provider
    if configurable.model_provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=reasoning_model,
            temperature=1.0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    elif configurable.model_provider == "ollama":
        if not configurable.ollama_api_base_url or not configurable.ollama_model_name:
            raise ValueError(
                "OLLAMA_API_BASE_URL and OLLAMA_MODEL_NAME must be set for Ollama provider"
            )
        llm = ChatOpenAI(
            model_name=configurable.ollama_model_name,
            openai_api_base=configurable.ollama_api_base_url,
            openai_api_key="NA",
            temperature=1.0,
        )
    elif configurable.model_provider == "azure_openai":
        if not configurable.azure_openai_api_version or \
           not configurable.azure_openai_deployment_name or \
           not configurable.azure_openai_api_base_url or \
           not configurable.azure_openai_api_key:
            raise ValueError(
                "Azure OpenAI configurations (API_VERSION, DEPLOYMENT_NAME, API_BASE_URL, API_KEY) must be set"
            )
        llm = AzureChatOpenAI(
            openai_api_version=configurable.azure_openai_api_version,
            azure_deployment=configurable.azure_openai_deployment_name,
            azure_endpoint=configurable.azure_openai_api_base_url,
            api_key=configurable.azure_openai_api_key,
            temperature=1.0,
        )
    else:
        raise ValueError(f"Unsupported model provider: {configurable.model_provider}")

    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model based on provider
    if configurable.model_provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=reasoning_model,
            temperature=0,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    elif configurable.model_provider == "ollama":
        if not configurable.ollama_api_base_url or not configurable.ollama_model_name:
            raise ValueError(
                "OLLAMA_API_BASE_URL and OLLAMA_MODEL_NAME must be set for Ollama provider"
            )
        llm = ChatOpenAI(
            model_name=configurable.ollama_model_name,
            openai_api_base=configurable.ollama_api_base_url,
            openai_api_key="NA",
            temperature=0,
        )
    elif configurable.model_provider == "azure_openai":
        if not configurable.azure_openai_api_version or \
           not configurable.azure_openai_deployment_name or \
           not configurable.azure_openai_api_base_url or \
           not configurable.azure_openai_api_key:
            raise ValueError(
                "Azure OpenAI configurations (API_VERSION, DEPLOYMENT_NAME, API_BASE_URL, API_KEY) must be set"
            )
        llm = AzureChatOpenAI(
            openai_api_version=configurable.azure_openai_api_version,
            azure_deployment=configurable.azure_openai_deployment_name,
            azure_endpoint=configurable.azure_openai_api_base_url,
            api_key=configurable.azure_openai_api_key,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported model provider: {configurable.model_provider}")

    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
