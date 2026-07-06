import sys
import os
from any_agent import AgentConfig, AnyAgent

def count_character_occurrences(word: str, char: str) -> int:
    """Count occurrences of a character in a word."""
    return word.count(char)

agent = AnyAgent.create(
    "openai",
    AgentConfig(
        model_id="openai:gpt-5",
        api_key=os.environ.get("OPENAI_API_KEY"),
        # model_id="llamafile:Qwen3.5-0.8B",
        # api_key="whatever",
        # api_base="http://localhost:8080",
        instructions="""You must use the available tools to find an answer.""",
        tools=[count_character_occurrences],
        callbacks=[]
    ),
)

char = 'r'
string = "strawberry"
# string = "strawrberrry"
# string = "Llanfair­pwllgwyngyll­gogery­chwyrn­drobwll­llan­tysilio­gogo­goch"
# string = "adsicnridsavubnsdrivubdsarridsbviusagrhadsifughrdsfiuhsda"
print(f"String: {string}\nOccurrences of {char}: {string.count(char)}")

prompt = sys.argv[1] if len(sys.argv) > 1 else f"How many times does the letter {char} occur in the word {string}?"
print(f"{prompt}\n")

for i in range(10):
    agent_trace = agent.run(prompt)
    print(f"Response #{i}: {agent_trace.final_output}")
