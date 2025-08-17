# mcp_server.py

from fastmcp import FastMCP
# from myapp.client_streamable_http import stream_data  # Import your function
from client_streamable_http import stream_data

# Create MCP server instance
mcp = FastMCP("MyProject MCP Server")

@mcp.tool
def run_stream():
    """Trigger the streamable HTTP client."""
    return stream_data()

if __name__ == "__main__":
    mcp.run()