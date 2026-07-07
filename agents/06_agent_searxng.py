# This agent uses SearXNG to simultaneously look for information on different
# search engines.

import httpx
import json
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
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        markdown_content = markdownify(response.text).strip()

        markdown_content = re.sub(r"\\n{2,}", "\\n", markdown_content)

        return str(markdown_content)

    except RequestException as e:
        return f"Error fetching the webpage: {e!s}"
    except Exception as e:
        return f"An unexpected error occurred: {e!s}"


def search(
    query: str,
    format: str = "json",
    categories: str = "",
    engines: str = "",
    language: str = "en",
    pageno: int = 1,
    time_range: str = "",
    safesearch: int = 1,
) -> str:
    """
    Search the web using SearXNG.

    Args:
        query: The search query (required)
        format: Output format (json, csv, rss) - default: json
        categories: Comma separated list of search categories (optional)
        engines: Comma separated list of search engines (optional)
        language: Language code - default: en
        pageno: Search page number - default: 1
        time_range: Time range (day, month, year) - optional
        safesearch: Safe search level (0, 1, 2) - default: 1
    """
    # put the URL where you can reach your SearXNG installation here
    search_url = "http://raspi:8888/search"

    params = {
        "q": query,
        "format": format,
        "language": language,
        "pageno": pageno,
        "safesearch": safesearch,
    }

    if categories:
        params["categories"] = categories
    if engines:
        params["engines"] = engines
    if time_range:
        params["time_range"] = time_range

    try:
        # TODO: Set user agent?
        headers = {}

        with httpx.Client(follow_redirects=True, timeout=30.0, headers=headers) as client:
            response = client.post(search_url, data=params)
            response.raise_for_status()

            if format == "json":
                result = response.json()
                if "results" in result:
                    formatted_results = []
                    for item in result["results"][:10]:
                        formatted_result = {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("content", ""),
                            "engine": item.get("engine", ""),
                        }
                        formatted_results.append(formatted_result)

                    summary = {
                        "query": query,
                        "number_of_results": len(result.get("results", [])),
                        "results": formatted_results,
                    }
                    return json.dumps(summary, indent=2)
                else:
                    return json.dumps(result, indent=2)
            else:
                return response.text

    except httpx.HTTPError as e:
        raise Exception(f"HTTP error occurred: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing JSON response: {e}")
    except Exception as e:
        raise Exception(f"Error performing search: {e}")


agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="llamafile:whatever",
        api_base="http://localhost:8080",
        instructions="""You must use the available tools to find an answer.""",
        tools=[search, visit_webpage],
    ),
)

prompt = sys.argv[1] if len(sys.argv) > 1 else """
Who are the speakers at the TUMO + mozilla.ai event in Yerevan?
"""
agent_trace = agent.run(prompt)

# also try: What are 5 tv shows that are trending in 2026? Please provide the name of the show, the exact release date, the genre, and a brief description of the show.