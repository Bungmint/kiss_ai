"""Configuration for IMO agent based on arXiv:2507.15855."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class IMOAgentConfig(BaseModel):
    model_name: str = Field(
        default="gemini-2.5-pro",
        description="LLM model to use (paper uses gemini-2.5-pro, grok-4, or gpt-5)",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature (paper: 0.1 for Gemini and Grok)",
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p sampling parameter",
    )
    thinking_budget: int = Field(
        default=32768,
        description="Max thinking tokens for Gemini models (paper: 32768)",
    )
    consecutive_passes_to_accept: int = Field(
        default=5,
        description="Accept solution after this many consecutive verification passes",
    )
    consecutive_errors_to_reject: int = Field(
        default=10,
        description="Reject solution after this many consecutive verification failures",
    )
    max_refinement_iterations: int = Field(
        default=30,
        description="Max verification-correction iterations per run",
    )
    max_runs: int = Field(
        default=10,
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
