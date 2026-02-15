"""Configuration for IMO agent based on arXiv:2507.15855."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class IMOAgentConfig(BaseModel):
    model_name: str = Field(
        default="gemini-3-pro-preview",
        description="Strong LLM for solving/correcting (needs deep reasoning)",
    )
    verifier_model_name: str = Field(
        default="gemini-2.5-pro",
        description="LLM for verification (needs good reasoning to check proofs)",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p sampling parameter",
    )
    thinking_budget: int = Field(
        default=32768,
        description="Max thinking tokens for solver",
    )
    verifier_thinking_budget: int = Field(
        default=16384,
        description="Max thinking tokens for verifier",
    )
    consecutive_passes_to_accept: int = Field(
        default=2,
        description="Accept solution after this many consecutive verification passes",
    )
    consecutive_errors_to_reject: int = Field(
        default=3,
        description="Reject solution after this many consecutive verification failures",
    )
    max_refinement_iterations: int = Field(
        default=8,
        description="Max verification-correction iterations per run",
    )
    max_runs: int = Field(
        default=3,
        description="Max independent solver attempts",
    )
    max_budget: float = Field(
        default=200.0,
        description="Maximum budget in USD",
    )


class IMOConfig(BaseModel):
    imo_agent: IMOAgentConfig = Field(
        default_factory=IMOAgentConfig,
        description="Configuration for IMO Agent",
    )


add_config("imo", IMOConfig)
