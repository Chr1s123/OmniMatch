def inject_preferences(prompt: str, preferences: list[str]) -> str:
    if not preferences:
        return prompt
    return f"{prompt}\n用户长期偏好：{', '.join(preferences)}"
