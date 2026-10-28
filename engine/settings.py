"""Runtime settings: the Anthropic key, model, effort, timeout, log level.

Read from environment variables, then from a local .env file (a variable already set in the
environment wins). The types do the validation: an unknown LLM_EFFORT or LOG_LEVEL fails at
startup with a clear message. The key is a SecretStr: it prints as ********** and has to be
unwrapped on purpose.

Paths are not settings: the data and the state file live in the repo. Business assumptions
(budgets, rules, thresholds) live in config.py.
"""

import logging
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STATE_PATH = ROOT / "state.json"


class Settings(BaseSettings):
    """The Anthropic key, model, effort, timeout and log level, from the environment or .env (the environment wins).

    A blank LLM_MODEL or LLM_EFFORT means the default (engine/llm.py), so `LLM_MODEL=` left empty in .env is fine.
    """

    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_effort: Literal["", "low", "medium", "high"] = ""
    llm_timeout_s: int = 120  # one attempt; the brief is one non-streamed reply, nothing arrives until it is complete
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def describe_llm(self) -> str:
        """Short label for the page, e.g. 'claude-sonnet-5-5 · effort medium'. Never includes the key."""
        if not self.anthropic_api_key.get_secret_value():
            return "none (no ANTHROPIC_API_KEY)"
        return f"{self.llm_model or 'claude-sonnet-5-5'} · effort {self.llm_effort or 'medium'}"


def get_settings() -> Settings:
    """Built on each call (it's cheap), so a shell or test that changes a variable is honoured."""
    return Settings()


def configure_logging(level: str | None = None) -> None:
    """Set up logging at the level from settings, or the one given."""
    logging.basicConfig(
        level=(level or get_settings().log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
