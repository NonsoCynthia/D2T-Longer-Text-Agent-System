__author__='chinonsocynthiaosuji'

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Utility functions for agent workflows, including variable substitution and step summarization in a uniquely formatted template.
"""

import re
from typing import List, Text, Union, Dict
from agents.utilities.utils import AgentStepOutput

def apply_variable_substitution(template: Text, substitutions: Union[Text, Dict[Text, Text]]) -> Text:
    """
    Substitute placeholders in the form {variable} within a template string.

    Args:
        template (Text): A string containing placeholders.
        substitutions (Union[Text, Dict[Text, Text]]): Substitution values.

    Returns:
        Text: The formatted string with placeholders replaced.
    """
    if isinstance(substitutions, str):
        # If a string, only support {user_prompt} placeholder
        return template.replace("{user_prompt}", substitutions)
    elif isinstance(substitutions, dict):
        # Replace all {var} with substitutions[var] or blank if missing
        def repl(match):
            var = match.group(1)
            return substitutions.get(var, "")
        pattern = re.compile(r"(?<!{){([^{}]+)}(?!})")
        return pattern.sub(repl, template)
    else:
        return template


def summarize_agent_steps(step_log: List[AgentStepOutput]) -> List[Text]:
    """
    Generate a uniquely formatted summary of agent execution steps,
    excluding guardrail steps and using UID-tagged blocks.

    Args:
        step_log (List[AgentStepOutput]): List of agent interaction records.

    Returns:
        List[Text]: A list of UID-formatted step summaries.
    """
    summary = []
    step_counter = 1

    for entry in step_log:
        agent = entry.agent_name.lower()

        if agent == "guardrail":
            continue

        if agent == "orchestrator":
            try:
                role, role_input = re.findall(r"(.*)\(input='(.*)'\)", entry.agent_output)[0]
            except Exception:
                role, role_input = "FINISH", entry.agent_output

            agent_type = "orchestrator"
            uid = f"{agent_type.upper()}_{step_counter}"
            if role == "FINISH":
                block = (
                    f"##=== BEGIN:{uid} ===##\n"
                    f"-- AGENT TYPE: {agent_type}\n"
                    f"-- AGENT NAME: {entry.agent_name}\n"
                    f"-- SIGNAL: FINISH\n"
                    f"-- RESPONSE START --\n{role_input}\n-- RESPONSE END --\n"
                    f"##=== END:{uid} ===##"
                )
            else:
                block = (
                    f"##=== BEGIN:{uid} ===##\n"
                    f"-- AGENT TYPE: {agent_type}\n"
                    f"-- AGENT NAME: {entry.agent_name}\n"
                    f"-- ROUTED TO: {role}\n"
                    f"-- INPUT START --\n{role_input}\n-- INPUT END --\n"
                    f"##=== END:{uid} ===##"
                )
        else:
            agent_type = entry.agent_name.lower()
            uid = f"{agent_type.upper()}_{step_counter}"

            if agent_type == "surface realization":
                block = (
                    f"##=== BEGIN:{uid} ===##\n"
                    f"-- AGENT TYPE: {agent_type}\n"
                    f"-- AGENT NAME: {entry.agent_name}\n"
                    f"-- INPUT START --\n{entry.agent_input}\n-- INPUT END --\n"
                    f"-- OUTPUT START --\n{entry.agent_output}\n-- OUTPUT END --\n"
                    "Finalizer Agent: Carefully review the output provided above by the surface realization agent. "
                    "Edit and refine the text as a human would, ensuring maximum fluency, semantic adequacy, coherence, and naturalness. "
                    "Your task is to produce the best possible final text, correcting any errors or awkwardness if present.\n"
                    f"##=== END:{uid} ===##"
                )
            else:
                block = (
                    f"##=== BEGIN:{uid} ===##\n"
                    f"-- AGENT TYPE: {agent_type}\n"
                    f"-- AGENT NAME: {entry.agent_name}\n"
                    f"-- INPUT START --\n{entry.agent_input}\n-- INPUT END --\n"
                    f"-- OUTPUT START --\n{entry.agent_output}\n-- OUTPUT END --\n"
                    f"##=== END:{uid} ===##"
                )

        summary.append(block)
        step_counter += 1

    return summary
