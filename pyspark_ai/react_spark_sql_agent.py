from typing import Any, Sequence, List

from langchain import BasePromptTemplate
from langchain.agents import Agent, AgentOutputParser
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.tools import BaseTool
from pydantic import Field

from pyspark_ai.prompt import (
    SPARK_SQL_PROMPT_VECTOR_SEARCH,
    SPARK_SQL_PROMPT_NO_VECTOR_SEARCH,
)
from pyspark_ai.tool import SimilarValueTool


class ReActSparkSQLAgent(Agent):
    """Agent for the ReAct chain."""

    output_parser: AgentOutputParser = Field(default_factory=MRKLOutputParser)

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return MRKLOutputParser()

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "ReActSparkSQL"

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Return default prompt."""
        for tool in tools:
            if isinstance(tool, SimilarValueTool):
                return SPARK_SQL_PROMPT_VECTOR_SEARCH
        return SPARK_SQL_PROMPT_NO_VECTOR_SEARCH

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def _stop(self) -> List[str]:
        return ["\nObservation:"]

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return "Thought:"

    @staticmethod
    def cache_key(desc: str, columns: str) -> str:
        """Return key for cache."""
        return f"sql transform desc:{desc}, columns:{columns}"
