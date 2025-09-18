__author__='chinonsocynthiaosuji'

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    This module defines utility classes and types used in the agent's execution pipeline.
"""

import re
from enum import Enum
from uuid import uuid4
from typing import Annotated, Dict, List, Text, Any, Optional, Union, TypedDict, Tuple
from pydantic import BaseModel, Field


class AgentStepOutput(BaseModel):
    """Internally used to store result steps in the agent's response."""
    agent_name: Text
    agent_input: Union[Text, Dict[str, Any]]
    agent_output: Union[List, Dict, Text]
    rationale: Optional[Union[List, Dict, Text]] = None

    
class ExecutionState(TypedDict, total=False):  # set total=False to make all keys optional
    """Holds evolving state across agent pipeline execution."""
    user_prompt: Union[Text, Dict[str, Any]]                    # User input to the agent system
    final_response: str                                         # Final agent response
    next_agent: str                                             # Next agent in the sequence
    next_agent_payload: str                                     # Inputs to the agents in the sequence
    review: str                                                 # Guardrail feedback
    iteration_count: int                                        # Count of all steps done
    max_iteration: int                                          # Recursion limit set in langChain
    current_step: int                                           # Current step count
    history_of_steps: List[AgentStepOutput]                     # A list of all the agent interactions
