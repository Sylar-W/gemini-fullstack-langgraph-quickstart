import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="gemini-2.0-flash",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash-preview-04-17",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="gemini-2.5-pro-preview-05-06",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    # Model Provider Configuration
    model_provider: str = Field(
        default="gemini",
        metadata={
            "description": "The model provider to use (e.g., 'gemini', 'ollama', 'azure_openai')."
        },
    )

    # Ollama Configuration
    ollama_api_base_url: Optional[str] = Field(
        default=None,
        metadata={"description": "The base URL for the Ollama API."},
    )
    ollama_model_name: Optional[str] = Field(
        default=None,
        metadata={"description": "The name of the Ollama model to use."},
    )

    # Azure OpenAI Configuration
    azure_openai_api_base_url: Optional[str] = Field(
        default=None,
        metadata={"description": "The base URL for the Azure OpenAI API."},
    )
    azure_openai_api_version: Optional[str] = Field(
        default=None,
        metadata={"description": "The API version for Azure OpenAI."},
    )
    azure_openai_deployment_name: Optional[str] = Field(
        default=None,
        metadata={"description": "The deployment name for Azure OpenAI."},
    )
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        metadata={"description": "The API key for Azure OpenAI."},
    )

    # Google Search Configuration
    google_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "API key for Google Search. If not provided, attempts to use GEMINI_API_KEY."
        },
    )
    google_cse_id: Optional[str] = Field(
        default=None,
        metadata={"description": "Google Custom Search Engine ID."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {}
        for name in cls.model_fields.keys():
            env_var_name = name.upper()
            # Special handling for google_api_key to also check GEMINI_API_KEY
            if name == "google_api_key":
                val = os.environ.get(
                    "GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY")
                )
            else:
                val = os.environ.get(env_var_name)

            if configurable.get(name) is not None:
                raw_values[name] = configurable.get(name)
            elif val is not None:
                raw_values[name] = val
            # Keep default if not in env or config

        # Filter out None values that were explicitly set (defaults are handled by Pydantic)
        # Pydantic will use field defaults if not provided in raw_values
        return cls(**raw_values)
