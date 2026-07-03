import yaml
from pathlib import Path

_ZERO_BOOT_CONFIG = Path(__file__).parent / "zero_boot_config.yaml"


def _load_zero_boot() -> set[str]:
    raw = yaml.safe_load(_ZERO_BOOT_CONFIG.read_text(encoding="utf-8"))
    return set(raw["zero_boot"])


def _resolve_index(cls, zero_boot: set[str], cache: dict) -> int:
    name = cls.__name__
    if name in cache:
        return cache[name]
    if name in zero_boot:
        cache[name] = 0
        return 0
    depends_on = getattr(cls, "depends_on", None)
    if not depends_on:
        raise RuntimeError(f"{name}: not in zero_boot_config and has no depends_on")
    index = max(_resolve_index(dep, zero_boot, cache) for dep in depends_on) + 1
    cache[name] = index
    return index


def _collect_dependencies(cls, zero_boot: set[str], registry: dict) -> None:
    if cls.__name__ in registry:
        return
    for dep in getattr(cls, "depends_on", []):
        _collect_dependencies(dep, zero_boot, registry)
        if dep.__name__ not in registry:
            registry[dep.__name__] = dep()
    registry[cls.__name__] = None  # placeholder; caller fills with actual instance


class ServiceManager:

    def __init__(self):
        self._zero_boot = _load_zero_boot()
        self._components: dict = {}  # name → instance

    def register(self, *components) -> None:
        for component in components:
            cls = type(component)
            _collect_dependencies(cls, self._zero_boot, self._components)
            self._components[cls.__name__] = component

    def start(self) -> None:
        cache = {}
        ordered = sorted(
            self._components.values(),
            key=lambda c: _resolve_index(type(c), self._zero_boot, cache),
        )
        for component in ordered:
            component.start()
            if hasattr(component, "ready_event"):
                component.ready_event.wait()

    def stop(self) -> None:
        cache = {}
        ordered = sorted(
            self._components.values(),
            key=lambda c: _resolve_index(type(c), self._zero_boot, cache),
            reverse=True,
        )
        for component in ordered:
            component.stop()
