"""Configuration Pydantic models for coding agent settings."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class RelentlessCodingAgentConfig(BaseModel):
    model_name: str = Field(
        default="claude-opus-4-6",
        description="LLM model to use",
    )
    max_steps: int = Field(
        default=25,
        description="Maximum steps for the Relentless Coding Agent",
    )
    max_budget: float = Field(
        default=200.0,
        description="Maximum budget in USD for the Relentless Coding Agent",
    )
    max_sub_sessions: int = Field(
        default=200,
        description="Maximum number of sub-sessions for auto-continuation",
    )


class EvolveOptimizerConfig(BaseModel):
    model_name: str = Field(
        default="claude-opus-4-6",
        description="LLM model for coder and monitor agents",
    )
    max_budget: float = Field(
        default=50.0,
        description="Maximum total coevolution budget in USD",
    )
    max_time: float = Field(
        default=3600.0,
        description="Maximum wall-clock time in seconds",
    )
    max_steps_per_session: int = Field(
        default=25,
        description="Maximum steps for each coder/monitor session",
    )
    max_sub_sessions: int = Field(
        default=10,
        description="Maximum sub-sessions per agent turn",
    )
    stop_on_target_score: bool = Field(
        default=False,
        description=(
            "Whether to stop the coevolution loop immediately when target_score is reached. "
            "If False, target_score is treated as a milestone and the run continues until "
            "budget/time/failure limits stop it."
        ),
    )


class CodingAgentConfig(BaseModel):
    relentless_coding_agent: RelentlessCodingAgentConfig = Field(
        default_factory=RelentlessCodingAgentConfig,
        description="Configuration for Relentless Coding Agent",
    )
    evolve_optimizer: EvolveOptimizerConfig = Field(
        default_factory=EvolveOptimizerConfig,
        description="Configuration for coevolving coder/monitor optimizer",
    )


# Register config with the global DEFAULT_CONFIG
add_config("coding_agent", CodingAgentConfig)
