"""MCP server package for bioeng-qpcr.

Exposes the MCPServer instance and a console entry point so the qPCR
analysis engine can be driven by MCP-compatible AI assistants.
"""

from bioeng_qpcr.mcp.server import main, mcp

__all__ = ["main", "mcp"]
