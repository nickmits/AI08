"""
LangGraph Client Agent for A2A Protocol Communication

This module implements a LangGraph-based client agent that can interact with
the A2A server. The client agent uses tools to make API calls to the server
and processes the responses intelligently.
"""

import logging
import os
from typing import Any, Annotated, TypedDict
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the state for our client agent
class ClientAgentState(TypedDict):
    """State for the client agent."""
    messages: Annotated[list, add_messages]
    a2a_task_id: str | None
    a2a_context_id: str | None
    base_url: str


# Global variables for A2A client (will be initialized when needed)
_httpx_client: httpx.AsyncClient | None = None
_a2a_client: A2AClient | None = None
_agent_card: AgentCard | None = None


async def initialize_a2a_client(base_url: str = "http://localhost:10000") -> A2AClient:
    """Initialize the A2A client with the agent card."""
    global _httpx_client, _a2a_client, _agent_card

    if _a2a_client is not None:
        return _a2a_client

    # Create httpx client with longer timeout
    _httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    # Initialize resolver
    resolver = A2ACardResolver(
        httpx_client=_httpx_client,
        base_url=base_url,
    )

    # Fetch the agent card
    logger.info(f"Fetching agent card from: {base_url}")
    _agent_card = await resolver.get_agent_card()
    logger.info("Successfully fetched agent card")

    # Create the A2A client
    _a2a_client = A2AClient(
        httpx_client=_httpx_client,
        agent_card=_agent_card
    )

    return _a2a_client


@tool
async def call_a2a_agent(
    query: str,
    task_id: str | None = None,
    context_id: str | None = None,
    base_url: str = "http://localhost:10000"
) -> dict[str, Any]:
    """
    Call the A2A agent with a query and get a response.

    This tool allows the client agent to communicate with the A2A server.
    It handles both initial queries and follow-up queries in multi-turn conversations.

    Args:
        query: The question or request to send to the A2A agent
        task_id: Optional task ID for continuing a conversation
        context_id: Optional context ID for continuing a conversation
        base_url: The base URL of the A2A server

    Returns:
        A dictionary containing the response and conversation IDs
    """
    logger.info(f"Calling A2A agent with query: {query}")

    # Initialize the client if needed
    client = await initialize_a2a_client(base_url)

    # Build the message payload
    send_message_payload: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [
                {'kind': 'text', 'text': query}
            ],
            'message_id': uuid4().hex,
        },
    }

    # Add task_id and context_id if this is a follow-up query
    if task_id and context_id:
        send_message_payload['message']['task_id'] = task_id
        send_message_payload['message']['context_id'] = context_id
        logger.info(f"Continuing conversation with task_id: {task_id}, context_id: {context_id}")

    # Create the request
    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(**send_message_payload)
    )

    # Send the request
    response = await client.send_message(request)

    # Extract the response data
    result = response.root.result
    response_message = result.message

    # Get the text content from the response
    text_content = ""
    if response_message and response_message.parts:
        for part in response_message.parts:
            if hasattr(part.root, 'text'):
                text_content += part.root.text

    # Also check artifacts for additional content
    artifacts_content = ""
    if result.artifacts:
        for artifact in result.artifacts:
            if artifact.parts:
                for part in artifact.parts:
                    if hasattr(part.root, 'text'):
                        artifacts_content += part.root.text

    # Combine content
    full_content = text_content + artifacts_content if artifacts_content else text_content

    logger.info(f"Received response from A2A agent: {full_content[:100]}...")

    return {
        "content": full_content,
        "task_id": result.id,
        "context_id": result.context_id,
        "status": result.status,
    }


async def cleanup_a2a_client():
    """Clean up the A2A client resources."""
    global _httpx_client, _a2a_client, _agent_card

    if _httpx_client:
        await _httpx_client.aclose()
        _httpx_client = None
        _a2a_client = None
        _agent_card = None


# Create the LangGraph client agent
def build_client_agent(base_url: str = "http://localhost:10000"):
    """
    Build a LangGraph-based client agent that can interact with the A2A server.

    Args:
        base_url: The base URL of the A2A server

    Returns:
        A compiled LangGraph graph
    """
    # Create the LLM
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Bind the A2A tool to the model
    tools = [call_a2a_agent]
    model_with_tools = model.bind_tools(tools)

    # Create tool node
    tool_node = ToolNode(tools)

    # Define the agent node
    def agent_node(state: ClientAgentState):
        """The main reasoning node for the client agent."""
        logger.info("Client agent reasoning about the query...")

        # Add system message if this is the first message
        messages = state["messages"]
        if len(messages) == 1 or not any(isinstance(m, AIMessage) for m in messages):
            system_message = HumanMessage(
                content=(
                    "You are a client agent that helps users by communicating with a remote A2A agent. "
                    "The remote agent has access to web search, academic paper search, and document retrieval tools. "
                    "When a user asks a question, use the call_a2a_agent tool to get information from the remote agent. "
                    f"The A2A server base URL is: {base_url}. "
                    "If you need to ask follow-up questions, use the task_id and context_id from previous responses. "
                    "Always provide the user's original query directly to the A2A agent - don't modify it."
                )
            )
            messages = [system_message] + messages

        # Call the model
        response = model_with_tools.invoke(messages)

        # Update state with task and context IDs if present
        new_state = {"messages": [response]}

        # Check if we have A2A IDs from tool responses in the messages
        for msg in reversed(state["messages"]):
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                # This is a simple approach - in production you'd parse the tool response more carefully
                if '"task_id"' in msg.content or '"context_id"' in msg.content:
                    # We have A2A IDs, they'll be available in the state
                    break

        return new_state

    # Define routing logic
    def should_continue(state: ClientAgentState):
        """Determine if we should continue to tools or end."""
        last_message = state["messages"][-1]

        # If there are tool calls, route to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"

        # Otherwise, end
        return END

    # Build the graph
    workflow = StateGraph(ClientAgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "agent")

    # Add memory for conversation persistence
    memory = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=memory)

    return app


async def run_client_agent_example():
    """
    Example function demonstrating how to use the client agent.
    """
    logger.info("=== Starting Client Agent Example ===")

    # Build the client agent
    base_url = "http://localhost:10000"
    client_agent = build_client_agent(base_url=base_url)

    # Create initial state
    initial_query = "What are the latest developments in artificial intelligence in 2025?"

    config = {"configurable": {"thread_id": "example-conversation-1"}}

    # Run the agent with the initial query
    logger.info(f"\n{'='*60}")
    logger.info(f"USER: {initial_query}")
    logger.info(f"{'='*60}\n")

    state = {
        "messages": [HumanMessage(content=initial_query)],
        "a2a_task_id": None,
        "a2a_context_id": None,
        "base_url": base_url
    }

    # Stream the agent's response
    async for event in client_agent.astream(state, config, stream_mode="values"):
        if "messages" in event:
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.content:
                logger.info(f"AGENT: {last_message.content}")

    # Clean up
    await cleanup_a2a_client()

    logger.info("\n=== Client Agent Example Complete ===")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_client_agent_example())
