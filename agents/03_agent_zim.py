# This example shows how third-party, local information can be used to improve
# model answers, without requiring the model to have this information in its
# training data.
# Check https://library.kiwix.org/ for ZIM files.
# (note: while the zim file is offline, uvx might need to be online to check for
# for the presence of some python deps)

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
                command="uvx",
                args=[
                    "zim-mcp-server",
                    "/Users/mala/Downloads/zim"
                ],
            )
        ],
    ),
)

prompt = sys.argv[1] if len(sys.argv) > 1 else """
When was Denny Vrandecic born?
"""
agent_trace = agent.run(prompt)
