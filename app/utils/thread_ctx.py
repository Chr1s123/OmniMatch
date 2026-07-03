def make_thread_id(raw: str) -> str:
    return raw if raw.startswith("thread_") else f"thread_{raw}"
