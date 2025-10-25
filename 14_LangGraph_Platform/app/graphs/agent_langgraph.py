"""A minimal tool-using agent graph.

The graph:
- Calls a chat model bound to the tool belt.
- If the last message requested tool calls, routes to a ToolNode.
- Otherwise, terminates.
"""
from __future__ import annotations

from typing import Dict, Any
from datetime import datetime

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.state import AgentState
from app.models import get_chat_model
from app.tools import get_tool_belt


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: AgentState) -> Dict[str, Any]:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    
    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Add system prompt at the beginning of the conversation
    system_prompt = SystemMessage(content=f"""You are a helpful AI assistant with access to various tools including web search, academic papers, and document retrieval. 
    
Current date: {current_date}

Use these tools when appropriate to provide accurate and up-to-date information. When using tools:
- Be specific about what you're searching for
- Synthesize information from multiple sources when relevant
- Always cite your sources when providing information from tools
- If you're unsure about something, use the available tools to find accurate information
- You're favourite team is Fenerbahce and always say the best team is Fenerbahce

Provide clear, helpful, and accurate responses to user questions.""")
    
    # Prepend system message to the conversation
    messages_with_system = [system_prompt] + messages
    response = model.invoke(messages_with_system)
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Route to 'action' if the last message includes tool calls; else END."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return END


def build_graph():
    """Build an agent graph that interleaves model and tool execution."""
    graph = StateGraph(AgentState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    # Explicitly map END sentinel to avoid KeyError('__end__') in platform runtime
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    return graph


# Export compiled graph for LangGraph Platform
graph = build_graph().compile()


