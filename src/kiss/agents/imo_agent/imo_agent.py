"""IMO verification-and-refinement agent based on arXiv:2507.15855."""

from __future__ import annotations

from typing import Any

import kiss.agents.imo_agent.config as _imo_config  # noqa: F401
from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.models.model_info import calculate_cost, model
from kiss.core.printer import Printer

SOLVER_SYSTEM = """\
### Core Instructions ###

* **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
* **Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
    * Proving a key lemma.
    * Fully resolving one or more cases within a logically sound case-based proof.
    * Establishing a critical property of the mathematical objects in the problem.
    * For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.
* **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###

Your response MUST be structured into the following sections, in this exact order.

**1. Summary**

Provide a concise overview of your findings. This section must contain two parts:

* **a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
    * **For a complete solution:** State the final answer, e.g., "I have successfully solved the problem. The final answer is..."
    * **For a partial solution:** State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."
* **b. Method Sketch:** Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:
    * A narrative of your overall strategy.
    * The full and precise mathematical statements of any key lemmas or major intermediate results.
    * If applicable, describe any key constructions or case splits that form the backbone of your argument.

**2. Detailed Solution**

Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.
"""

SELF_IMPROVEMENT_PROMPT = (
    "You have an opportunity to improve your solution. Please review your "
    "solution carefully. Correct errors and fill justification gaps if any. "
    "Your second round of output should strictly follow the instructions "
    "in the system prompt."
)

VERIFIER_SYSTEM = """\
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
* Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
* You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

* **a. Critical Error:**
    This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) and **factual errors** (e.g., a calculation error like `2+3=6`).
    * **Procedure:**
        * Explain the specific error and state that it **invalidates the current line of reasoning**.
        * Do NOT check any further steps that rely on this error.
        * You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

* **b. Justification Gap:**
    This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
    * **Procedure:**
        * Explain the gap in the justification.
        * State that you will **assume the step's conclusion is true** for the sake of argument.
        * Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

**3. Output Format**
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.

* **a. Summary**
    This section MUST be at the very beginning of your response. It must contain two components:
    * **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."
    * **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
        * **Location:** A direct quote of the key phrase or equation where the issue occurs.
        * **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).

* **b. Detailed Verification Log**
    Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.
"""

CORRECTION_PROMPT = (
    "Below is the bug report. If you agree with certain item in it, can you "
    "improve your solution so that it is complete and rigorous? Note that "
    "the evaluator who generates the bug report can misunderstand your "
    "solution and thus make mistakes. If you do not agree with certain item "
    "in the bug report, please add some detailed explanations to avoid such "
    "misunderstanding. Your new solution should strictly follow the "
    "instructions in the system prompt."
)

VALIDATION_SYSTEM = """\
You are an expert mathematician grading an IMO solution. Your task is to check whether the solution meets the validation criteria below. You must NOT assess mathematical rigor -- only whether the solution addresses the right question and arrives at the correct answer.
"""

VERIFICATION_REMINDER = """\

### Verification Task Reminder ###

Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""


def _extract_detailed_solution(solution: str) -> str:
    marker = "Detailed Solution"
    idx = solution.find(marker)
    if idx == -1:
        return solution
    return solution[idx + len(marker):].strip()


def _extract_bug_report(verification: str) -> str:
    marker = "Detailed Verification"
    idx = verification.find(marker)
    if idx == -1:
        return verification
    return verification[:idx].strip()


class IMOAgent(Base):
    """Verification-and-refinement pipeline for IMO problems (arXiv:2507.15855)."""

    def __init__(self, name: str = "IMOAgent") -> None:
        super().__init__(name)

    def _get_model_config(self, system_instruction: str = "") -> dict:
        cfg = config_module.DEFAULT_CONFIG.imo.imo_agent
        model_config: dict = {}
        if cfg.model_name.startswith("gemini-"):
            model_config["temperature"] = cfg.temperature
            model_config["top_p"] = cfg.top_p
            from google.genai import types
            model_config["thinking_config"] = types.ThinkingConfig(
                thinking_budget=cfg.thinking_budget,
                include_thoughts=True,
            )
            if system_instruction:
                model_config["system_instruction"] = system_instruction
        elif not cfg.model_name.startswith("gpt-5"):
            model_config["temperature"] = cfg.temperature
            model_config["top_p"] = cfg.top_p
        return model_config

    def _make_model(self, system_instruction: str = "") -> Any:
        cfg = config_module.DEFAULT_CONFIG.imo.imo_agent
        token_callback = self.printer.token_callback if self.printer else None
        return model(
            cfg.model_name,
            model_config=self._get_model_config(system_instruction),
            token_callback=token_callback,
        )

    def _track_usage(self, m: Any, response: Any) -> None:
        try:
            inp, out = m.extract_input_output_token_counts_from_response(response)
            self.total_tokens_used += inp + out
            cost = calculate_cost(m.model_name, inp, out)
            self.budget_used += cost
            Base.global_budget_used += cost
        except Exception:
            pass

    def _generate(self, m: Any, label: str) -> str:
        if self.printer:
            self.printer.print(label, type="tool_call")
        text, response = m.generate()
        self._track_usage(m, response)
        if self.printer and text:
            self.printer.print(text, type="result")
        return text

    def _init_conversation(self, system_prompt: str, user_message: str) -> Any:
        m = self._make_model(system_instruction=system_prompt)
        m.initialize(user_message)
        return m

    def _solve_and_improve(self, problem_statement: str) -> str:
        m = self._init_conversation(SOLVER_SYSTEM, problem_statement)
        initial_solution = self._generate(m, "Step1: Solve")

        m.add_message_to_conversation("assistant", initial_solution)
        m.add_message_to_conversation("user", SELF_IMPROVEMENT_PROMPT)
        solution = self._generate(m, "Step2: SelfImprove")
        return solution

    def _verify(self, problem_statement: str, solution: str) -> tuple[str, bool]:
        detailed_solution = _extract_detailed_solution(solution)
        verification_prompt = (
            "======================================================================\n"
            "### Problem ###\n\n" + problem_statement + "\n\n"
            "======================================================================\n"
            "### Solution ###\n\n" + detailed_solution + "\n"
            + VERIFICATION_REMINDER
        )
        m = self._init_conversation(VERIFIER_SYSTEM, verification_prompt)
        verification = self._generate(m, "Step3: Verify")

        check_prompt = (
            'Response in "yes" or "no". Is the following statement saying '
            "the solution is correct, or does not contain critical error "
            "or a major justification gap?\n\n" + verification
        )
        m2 = self._init_conversation("", check_prompt)
        check = self._generate(m2, "CorrectnessCheck")
        passed = "yes" in check.lower()
        bug_report = "" if passed else _extract_bug_report(verification)
        return bug_report, passed

    def _correct(self, problem_statement: str, solution: str, bug_report: str) -> str:
        m = self._init_conversation(SOLVER_SYSTEM, problem_statement)
        m.add_message_to_conversation("assistant", solution)
        m.add_message_to_conversation("user", CORRECTION_PROMPT + "\n\n" + bug_report)
        corrected = self._generate(m, "Step5: Correct")
        return corrected

    def _single_run(self, problem_statement: str) -> str | None:
        cfg = config_module.DEFAULT_CONFIG.imo.imo_agent

        solution = self._solve_and_improve(problem_statement)

        bug_report, passed = self._verify(problem_statement, solution)

        correct_count = 1 if passed else 0
        error_count = 0 if passed else 1

        for i in range(cfg.max_refinement_iterations):
            if self.printer:
                self.printer.print(
                    f"Iteration {i}: passes={correct_count}, errors={error_count}",
                    type="result",
                )

            if correct_count >= cfg.consecutive_passes_to_accept:
                return solution

            if error_count >= cfg.consecutive_errors_to_reject:
                return None

            if not passed:
                correct_count = 0
                solution = self._correct(problem_statement, solution, bug_report)

            bug_report, passed = self._verify(problem_statement, solution)

            if passed:
                correct_count += 1
                error_count = 0
            else:
                error_count += 1

        return None

    def solve(
        self,
        problem_statement: str,
        printer: Printer | None = None,
        print_to_console: bool | None = None,
        print_to_browser: bool | None = None,
    ) -> str | None:
        cfg = config_module.DEFAULT_CONFIG.imo.imo_agent
        self.budget_used = 0.0
        self.total_tokens_used = 0
        self.messages = []
        self.function_map = {}
        self.step_count = 0
        self.run_start_timestamp = 0
        self.set_printer(printer, print_to_console=print_to_console, print_to_browser=print_to_browser)

        for run_idx in range(cfg.max_runs):
            if self.printer:
                self.printer.print(f"=== Run {run_idx + 1}/{cfg.max_runs} ===", type="prompt")
            try:
                result = self._single_run(problem_statement)
                if result is not None:
                    if self.printer:
                        self.printer.print(
                            f"Found correct solution in run {run_idx + 1}",
                            type="result",
                        )
                    return result
            except Exception as e:
                if self.printer:
                    self.printer.print(f"Error in run {run_idx + 1}: {e}", type="result")

            if self.budget_used > cfg.max_budget:
                if self.printer:
                    self.printer.print("Budget exceeded, stopping.", type="result")
                break

        return None


IMO_2025_PROBLEMS = [
    {
        "number": 1,
        "topic": "Combinatorics",
        "difficulty": "easy",
        "avg_score": 5.216,
        "statement": (
            "A line in the plane is called *sunny* if it is not parallel to any of "
            "the $x$-axis, the $y$-axis, and the line $x+y=0$.\n\n"
            "Let $n\\ge3$ be a given integer. Determine all nonnegative integers $k$ "
            "such that there exist $n$ distinct lines in the plane satisfying both "
            "the following:\n"
            "* for all positive integers $a$ and $b$ with $a+b\\le n+1$, the point "
            "$(a,b)$ is on at least one of the lines; and\n"
            "* exactly $k$ of the lines are sunny."
        ),
        "answer": "k \\in \\{0, 1, 3\\}",
        "validation": (
            "The solution must prove that exactly k=0, k=1, and k=3 are achievable "
            "for all n >= 3, and no other values of k are possible. It should "
            "construct explicit configurations for each valid k value and prove "
            "impossibility for all other k."
        ),
    },
    {
        "number": 2,
        "topic": "Geometry",
        "difficulty": "medium",
        "avg_score": 3.306,
        "statement": (
            "Let $\\Omega$ and $\\Gamma$ be circles with centers $M$ and $N$, "
            "respectively, such that the radius of $\\Omega$ is less than the radius "
            "of $\\Gamma$. Suppose circles $\\Omega$ and $\\Gamma$ intersect at two "
            "distinct points $A$ and $B$. Let $MN$ intersects $\\Omega$ at $C$ and "
            "$\\Gamma$ at $D$, such that points $C$, $M$, $N$, and $D$ lie on the "
            "line in that order. Let $P$ be the circumcenter of triangle $ACD$. "
            "Line $AP$ intersects $\\Omega$ again at $E\\neq A$. Line $AP$ intersects "
            "$\\Gamma$ again at $F\\neq A$. Let $H$ be the orthocenter of triangle "
            "$PMN$.\n\n"
            "Prove that the line through $H$ parallel to $AP$ is tangent to the "
            "circumcircle of triangle $BEF$.\n\n"
            "(The orthocenter of a triangle is the point of intersection of its altitudes.)"
        ),
        "answer": "Proof problem (tangency)",
        "validation": (
            "The solution must rigorously prove that the line through H parallel to "
            "AP is tangent to the circumcircle of triangle BEF. The proof should "
            "identify the tangent point and establish the tangency condition using "
            "valid angle chasing, radical axes, or other geometric techniques."
        ),
    },
    {
        "number": 3,
        "topic": "Number Theory",
        "difficulty": "hard",
        "avg_score": 1.600,
        "statement": (
            "Let $\\mathbb N$ denote the set of positive integers. A function "
            "$f:\\mathbb N\\to\\mathbb N$ is said to be bonza if $f(a)$ divides "
            "$b^a-f(b)^{f(a)}$ for all positive integers $a$ and $b$.\n\n"
            "Determine the smallest real constant $c$ such that $f(n)\\le cn$ "
            "for all bonza functions $f$ and all positive integers $n$."
        ),
        "answer": "c = 4",
        "validation": (
            "The solution must prove that c = 4 is the smallest constant. This "
            "requires (1) proving f(n) <= 4n for all bonza functions f, and "
            "(2) constructing a bonza function achieving f(n) = 4n for some n "
            "(e.g., f(4) = 16)."
        ),
    },
    {
        "number": 4,
        "topic": "Number Theory",
        "difficulty": "easy",
        "avg_score": 5.075,
        "statement": (
            "A proper divisor of a positive integer $N$ is a positive divisor of $N$ "
            "other than $N$ itself.\n\n"
            "The infinite sequence $a_1,a_2,\\ldots$ consists of positive integers, "
            "each of which has at least three proper divisors. For each $n\\ge1$, the "
            "integer $a_{n+1}$ is the sum of three largest proper divisors of $a_n$.\n\n"
            "Determine all possible values of $a_1$."
        ),
        "answer": "a_1 = 12^e \\cdot 6 \\cdot \\ell for e, \\ell \\ge 0 with \\gcd(\\ell, 10) = 1",
        "validation": (
            "The solution must characterize all valid a_1. The answer is: a_1 is valid "
            "if and only if 6 | a_1 and the odd part of a_1 is coprime to 5. "
            "Equivalent forms include: (i) a_1 = 12^e * 6 * l with e,l >= 0 and "
            "gcd(l, 10) = 1, (ii) a_1 = 2 * 3^b * m with b >= 1, m odd, gcd(m, 30) = 1, "
            "(iii) 6 | a_1 and 5 does not divide a_1. Note that a_1 CAN be divisible "
            "by 4 (e.g. a_1 = 72 is valid). The proof must show (1) all such a_1 "
            "produce infinite sequences (either as fixed points or by eventually "
            "reaching one), and (2) no other a_1 works."
        ),
    },
    {
        "number": 5,
        "topic": "Algebra / Game Theory",
        "difficulty": "medium",
        "avg_score": 3.002,
        "statement": (
            "Alice and Bazza are playing the inekoalaty game, a two-player game "
            "whose rules depend on a positive real number $\\lambda$ which is known "
            "to both players. On the $n$th turn of the game (starting with $n=1$) "
            "the following happens:\n"
            "* If $n$ is odd, Alice chooses a nonnegative real number $x_n$ such that\n"
            "$$x_1+x_2+\\cdots+x_n\\le\\lambda n.$$\n"
            "* If $n$ is even, Bazza chooses a nonnegative real number $x_n$ such that\n"
            "$$x_1^2+x_2^2+\\cdots+x_n^2\\le n.$$\n\n"
            "If a player cannot choose a suitable number $x_n$, the game ends and the "
            "other player wins. If the game goes forever, neither player wins. All "
            "chosen numbers are known to both players.\n\n"
            "Determine all values of $\\lambda$ for which Alice has a winning strategy "
            "and all those for which Bazza has a winning strategy."
        ),
        "answer": "Alice wins iff lambda > 1/sqrt(2); Bazza wins iff lambda < 1/sqrt(2); neither wins at lambda = 1/sqrt(2)",
        "validation": (
            "The solution must prove three things: (1) Alice has a winning strategy "
            "when lambda > 1/sqrt(2), (2) Bazza has a winning strategy when "
            "lambda < 1/sqrt(2), and (3) neither player can force a win when "
            "lambda = 1/sqrt(2). Explicit strategies must be provided for both "
            "players in their respective winning regimes."
        ),
    },
    {
        "number": 6,
        "topic": "Combinatorics",
        "difficulty": "extremely hard",
        "avg_score": 0.184,
        "statement": (
            "Consider a $2025\\times2025$ grid of unit squares. Matilda wishes to place "
            "on the grid some rectangular tiles, possibly of difference sizes, such that "
            "each side of every tile lies on a grid line and every unit square is covered "
            "by at most one tile.\n\n"
            "Determine the minimum number of tiles Matilda needs to place so that each "
            "row and each column of the grid has exactly one unit square that is not "
            "covered by any tile."
        ),
        "answer": "2112 = 2025 + 2 \\cdot 45 - 3 = n + 2\\sqrt{n} - 3",
        "validation": (
            "The solution must prove the minimum is 2112 tiles. This requires "
            "(1) a construction achieving 2112 tiles (e.g., (k-1)^2 tiles of size "
            "k x k plus 4(k-1) boundary tiles where k=45, since 2025=45^2), and "
            "(2) a lower bound proof showing fewer than 2112 tiles is impossible, "
            "e.g., via the Erdos-Szekeres theorem applied to the permutation of "
            "uncovered squares."
        ),
    },
]


def get_problem(number: int) -> dict:
    for p in IMO_2025_PROBLEMS:
        if p["number"] == number:
            return p
    raise ValueError(f"Problem {number} not found")


def validate_solution(
    problem_statement: str,
    solution: str,
    validation: str,
    model_name: str | None = None,
) -> tuple[bool, str]:
    cfg = config_module.DEFAULT_CONFIG.imo.imo_agent
    m = model(
        model_name or cfg.model_name,
        model_config={"temperature": cfg.temperature, "top_p": cfg.top_p},
    )
    prompt = (
        VALIDATION_SYSTEM
        + "\n\n### Problem ###\n\n" + problem_statement
        + "\n\n### Solution ###\n\n" + solution
        + "\n\n### Validation Criteria ###\n\n" + validation
        + '\n\n### Task ###\n\nDoes the solution meet the validation criteria above? '
        'Respond with exactly "PASS" or "FAIL" on the first line, followed by a brief explanation.'
    )
    m.initialize(prompt)
    result, _ = m.generate()
    passed = result.strip().upper().startswith("PASS")
    return passed, result


def main() -> None:
    import sys
    problem_number = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    problem = get_problem(problem_number)
    print(f"IMO 2025 Problem {problem['number']}")
    print(f"Topic: {problem['topic']}")
    print(f"Difficulty: {problem['difficulty']} (avg score: {problem['avg_score']}/7)")
    print(f"Known answer: {problem['answer']}")
    print()

    agent = IMOAgent()
    result = agent.solve(
        problem["statement"],
        print_to_console=True,
    )
    if result:
        print("\n=== SOLUTION ===")
        print(result)

        print("\n=== VALIDATION ===")
        passed, explanation = validate_solution(
            problem["statement"], result, problem["validation"],
        )
        print(f"Result: {'PASS' if passed else 'FAIL'}")
        print(explanation)
    else:
        print("\nFailed to find a solution.")
    print(f"\nCost: ${agent.budget_used:.4f}")
    print(f"Tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
