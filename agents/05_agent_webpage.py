# A very simple agent which downloads a document from the Web and uses it to answer
# the user's question. Can follow links recursively.

import re
import requests
import sys

from any_agent import AgentConfig, AnyAgent
from markdownify import markdownify
from requests.exceptions import RequestException

def visit_webpage(url: str, timeout: int = 30) -> str:
    """Visits a webpage at the given url and returns its content as a markdown string. Use this to browse webpages.

    Args:
        url: The url of the webpage to visit.
        timeout: The timeout in seconds for the request.
    """
    headers = None
    try:
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()

        markdown_content = markdownify(response.text).strip()

        markdown_content = re.sub(r"\\n{2,}", "\\n", markdown_content)

        return str(markdown_content)

    except RequestException as e:
        return f"Error fetching the webpage: {e!s}"
    except Exception as e:
        return f"An unexpected error occurred: {e!s}"


agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="llamafile:Qwen3.5-9B-Q5_K_S",
        api_base="http://localhost:8080",
        instructions="""You must use the available tools to find an answer.""",
        tools=[visit_webpage],
    ),
)

prompt = sys.argv[1] if len(sys.argv) > 1 else """
What are the most recent two posts at http://aittalam.github.io about?
"""
agent_trace = agent.run(prompt)
