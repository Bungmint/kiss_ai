"""Repo agent that optimizes agent code using RelentlessCodingAgent."""

from __future__ import annotations

import argparse

from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

DEFAULT_MODEL = "claude-opus-4-6"

TASK_TEMPLATE = """
Your working directory is {work_dir}.

Can you run the command {command}
in the background so that you can monitor the output in real time,
and correct the code in the working directory if needed?  I MUST be able to 
see the command output in real time.

If you observe any repeated errors in the output or the command is not able
to complete successfully, please fix the code in the working directory and run the
command again.  Repeat the process until the command can finish successfully.

After the command finishes successfully, run the command again
and monitor its output in real time. You can add diagnostic code which will print 
metrics {metrics} information at finer level of granularity. 
Check for opportunities to optimize the code
on the basis of the metrics information---you need to minimize the metrics.   
If you discover any opportunities to minimize the metrics based on the code 
and the command output, optimize the code and run the command again.  
Repeat the process until the command can finish successfully,
and lower the metrics significantly.  Do not forget to remove the diagnostic 
code after the optimization is complete."""

def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a repository using RelentlessCodingAgent")
    parser.add_argument("--commnad", help="Command to run")
    parser.add_argument("--metrics", help="Metrics to optimize")
    parser.add_argument("--work-dir", default=".",
                        help=f"Working directory (default: .)")
    args = parser.parse_args()

    command = args.command
    metrics = args.metrics
    work_dir = args.work_dir

    task = TASK_TEMPLATE.format(command=command, metrics=metrics, work_dir=work_dir)
    agent = RelentlessCodingAgent("RepoOptimizer")
    result = agent.run(
        prompt_template=task,
        model_name=DEFAULT_MODEL,
        work_dir=work_dir
    )
    print(result)


if __name__ == "__main__":
    main()
