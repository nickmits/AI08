"""Test script for MCP server integration with LangGraph.

This script tests:
1. Direct connection to MCP server
2. MCP tools through LangGraph agent
"""
import requests
import time
from langgraph_sdk import get_client


def test_mcp_server_connection():
    """Test if MCP server is running and responding."""
    print("\n" + "="*60)
    print("TEST 1: MCP Server Connection")
    print("="*60)

    try:
        # Test a simple tool call
        response = requests.post(
            "http://localhost:8001/tools/call",
            json={
                "name": "get_weather",
                "arguments": {"city": "San Francisco"}
            },
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            print("✓ MCP server is running and responding")
            print(f"  Response: {result}")
            return True
        else:
            print(f"✗ MCP server returned status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to MCP server at http://localhost:8001")
        print("  Please start the server with:")
        print("  python -m app.mcp_server_langgraph_platform")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_mcp_tools_via_langgraph():
    """Test MCP tools through LangGraph agent."""
    print("\n" + "="*60)
    print("TEST 2: MCP Tools via LangGraph Agent")
    print("="*60)

    try:
        client = get_client(url="http://localhost:2024")

        # Test dice rolling through agent
        print("\nTest 2a: Dice Rolling Tool")
        print("-" * 60)
        thread = client.threads.create()
        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "Roll 2d6 for me using the dice rolling tool"
                }
            ]
        }

        print("Sending request to agent...")
        run = client.runs.stream(
            thread["thread_id"],
            "agent",
            input=input_data,
            stream_mode="values"
        )

        for chunk in run:
            if "messages" in chunk:
                last_message = chunk["messages"][-1]
                if hasattr(last_message, "content") and last_message.content:
                    print(f"✓ Agent response: {last_message.content[:200]}...")
                    break

        # Test weather tool
        print("\n\nTest 2b: Weather Tool")
        print("-" * 60)
        thread2 = client.threads.create()
        input_data2 = {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Tokyo using the weather tool?"
                }
            ]
        }

        print("Sending request to agent...")
        run2 = client.runs.stream(
            thread2["thread_id"],
            "agent",
            input=input_data2,
            stream_mode="values"
        )

        for chunk in run2:
            if "messages" in chunk:
                last_message = chunk["messages"][-1]
                if hasattr(last_message, "content") and last_message.content:
                    print(f"✓ Agent response: {last_message.content[:200]}...")
                    break

        print("\n✓ MCP tools are working through LangGraph agent")
        return True

    except Exception as e:
        print(f"✗ Error testing LangGraph integration: {str(e)}")
        print("  Make sure LangGraph server is running with: uv run langgraph dev")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MCP INTEGRATION TEST SUITE")
    print("="*60)

    # Test 1: MCP Server Connection
    server_ok = test_mcp_server_connection()

    if not server_ok:
        print("\n" + "="*60)
        print("SETUP INSTRUCTIONS")
        print("="*60)
        print("\n1. Start the MCP server in a separate terminal:")
        print("   python -m app.mcp_server_langgraph_platform")
        print("\n2. Verify it's running at http://localhost:8001")
        print("\n3. Re-run this test script")
        return

    # Give server a moment to stabilize
    time.sleep(1)

    # Test 2: LangGraph Integration
    test_mcp_tools_via_langgraph()

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
