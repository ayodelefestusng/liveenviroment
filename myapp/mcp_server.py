# mcp_server.py

# from myapp.client_streamable_http import stream_data  # Import your function
from client_streamable_http import stream_data
from fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("MyProject MCP Server")

@mcp.tool
def run_stream():
    """Trigger the streamable HTTP client."""
    return stream_data()

if __name__ == "__main__":
    mcp.run()