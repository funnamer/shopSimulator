"""Translate standard OpenAI tool calls to the unchanged ShopEnv API."""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


SHOP_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the shopping catalog when search is available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Precise product search keywords.",
                    }
                },
                "required": ["keywords"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click one exact value from the current clickable-buttons list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "Exact value from the current clickable-buttons list.",
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    },
]

ASK_SHOPPER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_shopper",
        "description": "Ask the shopper one question to clarify their requirements.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A natural-language question for the shopper.",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class ToolExecution:
    """One executed tool call and its standard role=tool response."""

    name: str
    output: Dict[str, Any]
    message: Dict[str, Any]


def assistant_message_to_dict(message: Any) -> Dict[str, Any]:
    """Convert an OpenAI SDK message, or compatible dict, for message history."""
    if isinstance(message, dict):
        result = dict(message)
    elif hasattr(message, "model_dump"):
        result = message.model_dump(exclude_none=True)
    else:
        raise TypeError(f"Unsupported assistant message type: {type(message)!r}")
    result.setdefault("role", "assistant")
    return result


class ShopToolAdapter:
    """Execute model tool calls without changing the environment implementation."""

    def __init__(
        self,
        shop_env: Any,
        ask_shopper: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.shop_env = shop_env
        self.ask_shopper = ask_shopper

    @property
    def tools(self) -> List[Dict[str, Any]]:
        tools = list(SHOP_TOOLS)
        if self.ask_shopper is not None:
            tools.append(ASK_SHOPPER_TOOL)
        return tools

    def execute_assistant_message(self, message: Dict[str, Any]) -> ToolExecution:
        """Validate and execute exactly one standard Chat Completions tool call."""
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) != 1:
            raise ValueError(
                f"Agent must make exactly one tool call per turn; got {len(tool_calls)}"
            )

        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            if hasattr(tool_call, "model_dump"):
                tool_call = tool_call.model_dump(exclude_none=True)
            else:
                raise TypeError(f"Unsupported tool call type: {type(tool_call)!r}")

        function = tool_call.get("function") or {}
        name = function.get("name")
        call_id = tool_call.get("id")
        if not name or not call_id:
            raise ValueError("Tool call is missing function.name or id")

        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = (
                raw_arguments
                if isinstance(raw_arguments, dict)
                else json.loads(raw_arguments)
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON arguments for tool {name}") from exc

        output = self._execute(name, arguments)
        return ToolExecution(
            name=name,
            output=output,
            message={
                "role": "tool",
                "tool_call_id": call_id,
                # Only feed the model the observation it needs for its next
                # decision.  Structured execution metadata remains available
                # through ToolExecution.output for control flow/diagnostics.
                "content": self._message_content(output),
            },
        )

    @staticmethod
    def _message_content(output: Dict[str, Any]) -> str:
        if output.get("source") == "environment":
            if output.get("ok", False):
                return str((output.get("result") or {}).get("instruction", ""))
            return str(output.get("error", "Environment error"))
        if output.get("source") == "shopper":
            return str(output.get("response", ""))
        return ""

    def _execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "search":
            keywords = self._required_string(arguments, "keywords", name)
            return self._execute_env(f"search[{keywords}]")
        if name == "click":
            value = self._required_string(arguments, "value", name)
            return self._execute_env(f"click[{value}]")
        if name == "ask_shopper" and self.ask_shopper is not None:
            question = self._required_string(arguments, "question", name)
            return {
                "ok": True,
                "source": "shopper",
                "response": self.ask_shopper(question),
            }
        raise ValueError(f"Unsupported tool: {name}")

    def _execute_env(self, action: str) -> Dict[str, Any]:
        result = self.shop_env.interact(action)
        if "error" in result:
            return {
                "ok": False,
                "source": "environment",
                "action": action,
                "error": result["error"],
            }
        return {
            "ok": True,
            "source": "environment",
            "action": action,
            "result": result,
        }

    @staticmethod
    def _required_string(arguments: Dict[str, Any], key: str, tool_name: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Tool {tool_name} requires a non-empty string '{key}'")
        return value.strip()
