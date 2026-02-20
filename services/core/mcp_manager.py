# Stubbed MCP Manager to prevent startup crashes
import os, asyncio
class MCPClientManager:
    def __init__(self): self.tools = []
    async def connect(self): logger.info("🐙 MCP: Connecting (Mock)...")
    async def cleanup(self): pass
mcp_manager = MCPClientManager()
