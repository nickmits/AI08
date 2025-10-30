"""
Example: Using the LangGraph Client Agent to Interact with A2A Server

This script demonstrates Activity #1: Building a LangGraph agent that uses
the A2A protocol to communicate with the server agent.

Before running this script:
1. Start the A2A server: uv run python -m app
2. Ensure the server is running on http://localhost:10000
3. Run this script: uv run python example_client_agent.py
"""

import asyncio
import logging
from langchain_core.messages import HumanMessage

from app.client_agent_langgraph import (
    build_client_agent,
    cleanup_a2a_client
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def single_query_example():
    """
    Example 1: Single query to the A2A agent
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 1: Single Query")
    logger.info("="*80 + "\n")

    # Build the client agent
    client_agent = build_client_agent(base_url="http://localhost:10000")

    # Create a query
    query = "Find me recent papers on transformer architectures from 2024"

    # Create initial state
    state = {
        "messages": [HumanMessage(content=query)],
        "a2a_task_id": None,
        "a2a_context_id": None,
        "base_url": "http://localhost:10000"
    }

    config = {"configurable": {"thread_id": "example-1"}}

    logger.info(f"👤 USER: {query}\n")

    # Run the agent
    final_state = None
    async for event in client_agent.astream(state, config, stream_mode="values"):
        if "messages" in event:
            final_state = event

    # Print the final response
    if final_state and final_state["messages"]:
        last_message = final_state["messages"][-1]
        logger.info(f"\n🤖 AGENT RESPONSE:\n{last_message.content}\n")


async def multi_turn_conversation_example():
    """
    Example 2: Multi-turn conversation with the A2A agent
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: Multi-Turn Conversation")
    logger.info("="*80 + "\n")

    # Build the client agent
    client_agent = build_client_agent(base_url="http://localhost:10000")

    # Use the same thread_id for conversation continuity
    config = {"configurable": {"thread_id": "multi-turn-example"}}

    # First query
    query1 = "What are the main trends in AI for 2025?"

    state1 = {
        "messages": [HumanMessage(content=query1)],
        "a2a_task_id": None,
        "a2a_context_id": None,
        "base_url": "http://localhost:10000"
    }

    logger.info(f"👤 USER (Turn 1): {query1}\n")

    final_state1 = None
    async for event in client_agent.astream(state1, config, stream_mode="values"):
        if "messages" in event:
            final_state1 = event

    if final_state1 and final_state1["messages"]:
        last_message = final_state1["messages"][-1]
        logger.info(f"🤖 AGENT (Turn 1):\n{last_message.content}\n")

    # Second query (follow-up)
    await asyncio.sleep(2)  # Brief pause between turns

    query2 = "Can you tell me more about the first trend you mentioned?"

    state2 = {
        "messages": [HumanMessage(content=query2)],
        "a2a_task_id": final_state1.get("a2a_task_id"),
        "a2a_context_id": final_state1.get("a2a_context_id"),
        "base_url": "http://localhost:10000"
    }

    logger.info(f"👤 USER (Turn 2): {query2}\n")

    final_state2 = None
    async for event in client_agent.astream(state2, config, stream_mode="values"):
        if "messages" in event:
            final_state2 = event

    if final_state2 and final_state2["messages"]:
        last_message = final_state2["messages"][-1]
        logger.info(f"🤖 AGENT (Turn 2):\n{last_message.content}\n")


async def test_different_tool_types():
    """
    Example 3: Test queries that trigger different tools on the server
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 3: Testing Different Tool Types")
    logger.info("="*80 + "\n")

    # Build the client agent
    client_agent = build_client_agent(base_url="http://localhost:10000")

    # Test cases for different tools
    test_queries = [
        ("Web Search (Tavily)", "What are the latest AI news from this week?"),
        ("Academic Papers (ArXiv)", "Find papers on multimodal transformers"),
        ("Document Retrieval (RAG)", "What information is in the documents about AI usage?"),
    ]

    for idx, (tool_name, query) in enumerate(test_queries, 1):
        logger.info(f"\n--- Test {idx}: {tool_name} ---")
        logger.info(f"👤 USER: {query}\n")

        state = {
            "messages": [HumanMessage(content=query)],
            "a2a_task_id": None,
            "a2a_context_id": None,
            "base_url": "http://localhost:10000"
        }

        config = {"configurable": {"thread_id": f"test-{idx}"}}

        final_state = None
        async for event in client_agent.astream(state, config, stream_mode="values"):
            if "messages" in event:
                final_state = event

        if final_state and final_state["messages"]:
            last_message = final_state["messages"][-1]
            logger.info(f"🤖 AGENT: {last_message.content[:200]}...\n")

        # Brief pause between queries
        await asyncio.sleep(1)


async def main():
    """
    Main function to run all examples
    """
    logger.info("\n" + "🚀 "*20)
    logger.info("LangGraph Client Agent - A2A Protocol Examples")
    logger.info("🚀 "*20 + "\n")

    logger.info("""
    Building a LangGraph agent that communicates with the A2A server.

    The client agent:
    - Uses LangGraph for structured reasoning
    - Has a tool (call_a2a_agent) to interact with the A2A server
    - Maintains conversation state across multiple turns
    - Routes queries intelligently through the A2A protocol
    """)

    try:
        # Run Example 1: Single query
        await single_query_example()

        # Wait a bit between examples
        await asyncio.sleep(2)

        # Run Example 2: Multi-turn conversation
        # Uncomment to run this example:
        # await multi_turn_conversation_example()

        # Wait a bit between examples
        # await asyncio.sleep(2)

        # Run Example 3: Different tool types
        # Uncomment to run this example:
        # await test_different_tool_types()

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)

    finally:
        # Clean up resources
        await cleanup_a2a_client()
        logger.info("\n✅ Examples complete! Client resources cleaned up.\n")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
