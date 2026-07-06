import sys
import os
from any_agent import AgentConfig, AnyAgent
from pathlib import Path

def read_file(file_name: str) -> str:
    """Read the contents of the given `file_name`.

    Args:
        file_name: The path to the file you want to read.

    Returns:
        The contents of `file_name`.

    Raises:
        ValueError: For the following cases:
            - If the path to the file is not allowed.
    """
    file_path = Path(file_name)
    return file_path.read_text()


def scan_current_dir(pattern: str) -> list[str]:
    """Scans the current directory for files satisfying the provided pattern.
    
    Args:
        pattern: The pattern used to filter files in the current directory (e.g. "*.txt"
        for text files, "*.py" for python files, "*.*" for all files)

    Returns:
        A string representing the list of filenames that satisfy the provided pattern
    """
    current_dir = Path(".")
    files_list = [str(f) for f in current_dir.glob(pattern)]
    return str(files_list)


agent = AnyAgent.create(
    "openai",
    AgentConfig(
        model_id="openai:gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY"),
        # model_id="llamafile:Qwen3.5-0.8B",
        # api_key="whatever",
        # api_base="http://localhost:8080",
        instructions="""You must use the available tools to find an answer.""",
        tools=[scan_current_dir, read_file],
        # callbacks=[],
    ),
)

prompt = sys.argv[1] if len(sys.argv) > 1 else "When was Davide Eynard born?"
agent_trace = agent.run(prompt)
print(agent_trace.final_output)

