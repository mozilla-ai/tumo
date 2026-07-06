# This agent uses the Joplin MCP server to connect to my Joplin notes.
# It is 100% local and allows me to easily find information I previously noted down.
# This is particularly useful in conjunction with "LLM wikis" (see
# https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

import sys
from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPStdio

agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="llamafile:Qwen3.5-9B-Q5_K_S",
        api_base="http://localhost:8080",
        instructions="""You must use the available tools to find an answer.""",
        tools=[
            MCPStdio(
                command="uv",
                args=[
                    "--directory",
                    "/Users/mala/workspace/joplin-mcp-server",
                    "run",
                    "src/mcp/joplin_mcp.py"
                ],
            )
        ],
    ),
)

prompt = sys.argv[1] if len(sys.argv) > 1 else """
Look into morla wiki and tell me which are the main issues I fixed wrt llamafile GPU acceleration.
"""
agent_trace = agent.run(prompt)