# This example shows how third-party, local information can be used to improve
# model answers, without requiring the model to have this information in its
# training data.
# Check https://library.kiwix.org/ for ZIM files.

import subprocess
import sys
from any_agent import AgentConfig, AnyAgent
from pathlib import Path


def find_zim_files(zim_files_dir: str = "~/Downloads/zim/", pattern: str = "*.zim") -> list[str]:
    """Scans the zim_files_dir directory for zim files.
    
    Args:
        zim_files_dir: the directory where zim files are stored
        pattern: The pattern used to filter files in the current directory

    Returns:
        A string representing the list of zim files in the provided directory.

    NOTE: Use args defaults if no values are explicitly provided
    """
    
    current_dir = Path(zim_files_dir).expanduser()
    files_list = [str(f) for f in current_dir.glob(pattern)]
    return str(files_list)


def call_zim_cli(zim_file: str, query: str) -> str:
    """Search a ZIM file by calling the zim_utils CLI.

    Args:
        zim_file: Path to the .zim file.
        query: The search query string.

    Returns:
        The CLI output as a string.
    """
    result = subprocess.run(
        [sys.executable, "-m", "zim_utils", zim_file, query],
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else result.stderr 


agent = AnyAgent.create(
    "openai",
    AgentConfig(
        model_id="llamafile:whatever",
        api_base="http://localhost:8080",
        instructions="""You must use the available tools to find an answer.""",
        tools=[find_zim_files, call_zim_cli],
    ),
)

prompt = sys.argv[1] if len(sys.argv) > 1 else """
When was Denny Vrandecic born?
"""
agent_trace = agent.run(prompt)

# Also try:
# "When are two triangles congruent? Use the tools and provide references for your answer"

# ZIM files used:
# https://mirrors.dotsrc.org/kiwix/zim/wikipedia/wikipedia_en_all_mini_2026-06.zim (11.7GB)
# https://mirrors.dotsrc.org/kiwix/zim/libretexts/libretexts.org_en_math_2026-01.zim (792MB)
