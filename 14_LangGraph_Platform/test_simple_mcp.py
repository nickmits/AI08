"""Simple test for MCP server integration with fetch_goout_events tool."""
import requests
from langgraph_sdk import get_client


def test_mcp_server():
    """Test if MCP server is running."""
    print("\n" + "="*60)
    print("TEST 1: MCP Server Connection")
    print("="*60)

    try:
        response = requests.post(
            "http://localhost:8002/tools/call",
            json={
                "name": "fetch_goout_events",
                "arguments": {"category": "nightlife", "days_ahead": 7}
            },
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("[OK] MCP server is running")
            print(f"[OK] Tool response: {result.get('content', [{}])[0].get('text', '')[:100]}...")
            return True
        else:
            print(f"[FAIL] MCP server returned status {response.status_code}")
            return False

    except Exception as e:
        print(f"[FAIL] Error: {str(e)}")
        return False


def test_langgraph_integration():
    """Test MCP tool through LangGraph agent."""
    print("\n" + "="*60)
    print("TEST 2: LangGraph Agent with MCP Tool")
    print("="*60)

    try:
        client = get_client(url="http://localhost:2024")
        thread = client.threads.create()

        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "Find nightlife events in the next 7 days using the goout events tool"
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
                    print(f"[OK] Agent response: {last_message.content[:200]}...")
                    break

        print("\n[OK] LangGraph integration working!")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SIMPLE MCP INTEGRATION TEST")
    print("="*60)

    # Test MCP server
    if test_mcp_server():
        # Test LangGraph integration
        test_langgraph_integration()

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
