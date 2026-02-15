"""Repo agent that solves tasks in the current project root using RelentlessCodingAgent."""

from __future__ import annotations

from pathlib import Path

from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

PROJECT_ROOT = str(Path(__file__).resolve().parents[4])


task = """
can you create an agent in src/kiss/agents/imo_agent/imo_agent.py using 
src/kiss/agents/coding_agents/relentless_coding_agent.py based on the agent 
described in the paper https://arxiv.org/abs/2507.15855 .  
The agent should be run on all the IMO 2025 tasks, where each task is a string 
consisting of an actual IMO 2025 problem statement followed by the criterion 
to validate the results.  Run the agent on the simplest task of all the tasks from IMO 2025.
"""

def main() -> None:
    agent = RelentlessCodingAgent("IMOAgentCreator")
    result = agent.run(
        prompt_template=task,
        model_name="claude-opus-4-6",
        work_dir=PROJECT_ROOT
    )
    print(result)


if __name__ == "__main__":
    main()
