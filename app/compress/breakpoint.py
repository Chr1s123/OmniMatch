def choose_breakpoint(messages: list[dict], keep_last: int = 6) -> int:
    return max(0, len(messages) - keep_last)
