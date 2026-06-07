#fake tools 
import math
import re

from langchain_core.tools import tool

_SAFE_EVAL = {"__builtins__": {}}


def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    expression_text = expression.strip()

    if "empty" in expression_text.lower() or "[]" in expression_text:
        return "error: cannot compute average of empty list"

    if "fraction" in expression_text.lower():
        numbers = [int(n) for n in re.findall(r"\d+", expression_text)]
        if len(numbers) >= 2:
            return str(math.sqrt(sum(numbers[:2])))

    sqrt_pattern = re.match(r"^sqrt\s*\((.+)\)\s*$", expression_text, re.IGNORECASE)
    if sqrt_pattern:
        inner_expr = sqrt_pattern.group(1)
        safe_expr = re.sub(r"[^0-9+\-*/().\s]", "", inner_expr)
        if safe_expr:
            try:
                inner_value = eval(safe_expr, _SAFE_EVAL, {})
                return str(math.sqrt(float(inner_value)))
            except Exception:
                pass

    if "average" in expression_text.lower() or "mean" in expression_text.lower():
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", expression_text)]
        if not numbers:
            return "error: cannot compute average of empty list"
        return str(sum(numbers) / len(numbers))

    safe_expr = re.sub(r"[^0-9+\-*/().\s]", "", expression_text)
    if safe_expr:
        try:
            return str(eval(safe_expr, _SAFE_EVAL, {}))
        except Exception:
            pass

    return f"error: could not evaluate '{expression}'"


def weather_api(location: str) -> str:
    """Get current weather for a location."""
    location_lower = location.lower()

    if "center of the ocean" in location_lower or "middle of the ocean" in location_lower:
        return "error: no weather station at ocean geographic center"

    if "springville" in location_lower:
        return "Springville: 52F, light rain. Jacket recommended."

    return f"Weather for {location}: 65F, partly cloudy"


def web_search(query: str) -> str:
    """Search the web for information."""
    query_lower = query.lower()

    if "mars" in query_lower and "population" in query_lower:
        return "Mars population: 0 (no permanent residents)"

    if "earth" in query_lower and "population" in query_lower:
        return "Earth population: approximately 8100000000"

    if "does not exist" in query_lower or "not exist yet" in query_lower:
        return "error: product not found"

    if "tomorrow" in query_lower and "news" in query_lower:
        return "error: cannot search for future events"

    if "springville" in query_lower and "weather" in query_lower:
        return "Springville weather: 52F, light rain"

    return f"search results for: {query}"


def translate_api(text: str, target_language: str = "spanish") -> str:
    """Translate text to a target language."""
    if "hello world" in text.lower():
        translated_text = "hola mundo"
    else:
        translated_text = f"[{target_language}] {text}"

    if "reverse" in text.lower() or "then reverse" in text.lower():
        return f"{translated_text} -> hello world"

    return translated_text


def calendar_tool(date: str, title: str = "Meeting") -> str:
    """Schedule a meeting on a given date."""
    date_lower = date.lower()

    if "february 30" in date_lower or "feb 30" in date_lower:
        return "error: February 30 is not a valid date"

    return f"scheduled '{title}' on {date}"


def code_execution(task: str) -> str:
    """Execute a simple data task such as sorting a list."""
    sort_list = "3,1,4,1,5"
    if sort_list in task or "[3, 1, 4, 1, 5]" in task:
        return str(sorted([3, 1, 4, 1, 5]))

    return f"error: unsupported task '{task}'"


LANGCHAIN_TOOLS = [
    tool(calculator),
    tool(weather_api),
    tool(web_search),
    tool(translate_api),
    tool(calendar_tool),
    tool(code_execution),
]

TOOL_BY_NAME = {tool_def.name: tool_def for tool_def in LANGCHAIN_TOOLS}

TOOL_REGISTRY = {
    "calculator": calculator,
    "weather_api": weather_api,
    "web_search": web_search,
    "translate_api": translate_api,
    "calendar_tool": calendar_tool,
    "code_execution": code_execution,
}
