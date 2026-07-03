class PreferenceStore:
    def __init__(self) -> None:
        self._preferences: list[str] = []

    def add(self, preference: str) -> None:
        self._preferences.append(preference)

    def list(self) -> list[str]:
        return list(self._preferences)
