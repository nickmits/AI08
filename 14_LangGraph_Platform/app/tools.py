"""Toolbelt assembly for agents.

Collects third-party tools and local tools (like RAG) into a single list that
graphs can bind to their language models.
"""
from __future__ import annotations

from typing import List
import requests
import json

from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from app.rag import retrieve_information

# MCP Server Configuration
MCP_SERVER_URL = "http://localhost:8002"


def _call_mcp_server(tool_name: str, arguments: dict) -> str:
    """
    Call MCP server tool via simple HTTP endpoint.

    Args:
        tool_name: Name of the MCP tool to call
        arguments: Dictionary of arguments to pass to the tool

    Returns:
        Result from the MCP server tool
    """
    try:
        payload = {
            "tool_name": tool_name,
            "arguments": arguments
        }

        response = requests.post(
            f"{MCP_SERVER_URL}/call_tool",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("result", "No result returned")
            else:
                return f"[MCP Error] {result.get('error', 'Unknown error')}"
        else:
            return f"[MCP Error] Server returned status {response.status_code}"

    except requests.exceptions.ConnectionError:
        return f"[MCP Error] Cannot connect to MCP server at {MCP_SERVER_URL}. Make sure it's running on port 8002."
    except requests.exceptions.Timeout:
        return f"[MCP Error] Request to MCP server timed out"
    except Exception as e:
        return f"[MCP Error] Failed to call MCP server: {str(e)}"


@tool
def mcp_fetch_goout_events(
    category: str = "nightlife",
    location: str = None,
    start_date: str = None,
    end_date: str = None,
    days_ahead: int = 30
) -> str:
    """
    Fetch events from GO-OUT.co API via MCP server. Great for finding nightlife, concerts, sports events.

    Args:
        category: Event category (nightlife, concerts, sports, all)
        location: Location filter (optional)
        start_date: Event start date in YYYY-MM-DD format (optional)
        end_date: Event end date in YYYY-MM-DD format (optional)
        days_ahead: Number of days ahead if start_date/end_date not provided (default: 30)

    Returns:
        Formatted string with event listings from the MCP server
    """
    arguments = {
        "category": category,
        "days_ahead": days_ahead
    }

    # Only include optional parameters if provided
    if location:
        arguments["location"] = location
    if start_date:
        arguments["start_date"] = start_date
    if end_date:
        arguments["end_date"] = end_date

    return _call_mcp_server("fetch_goout_events", arguments)




def get_tool_belt() -> List:
    """Return the list of tools available to agents (Arxiv, RAG, MCP tool)."""
    return [
        ArxivQueryRun(),
        retrieve_information,
        mcp_fetch_goout_events
    ]


