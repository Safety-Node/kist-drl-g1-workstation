def _resolve_index(cls, cache: dict) -> int:
    name = cls.__name__
    if name in cache:
        return cache[name]
    depends_on = getattr(cls, "depends_on", None)
    if depends_on is None:
        raise RuntimeError(f"{cls.__name__}: depends_on not declared")
    if len(depends_on) == 0:
        cache[name] = 0
        return 0
    index = max(_resolve_index(dep, cache) for dep in depends_on) + 1
    cache[name] = index
    return index


def _collect_dependencies(cls, registry: dict) -> None:
    if cls.__name__ in registry:
        return
    for dep in getattr(cls, "depends_on", []):
        _collect_dependencies(dep, registry)
        if dep.__name__ not in registry:
            registry[dep.__name__] = dep()
    registry[cls.__name__] = None  # placeholder; caller fills with actual instance


class ServiceManager:

    def __init__(self):
        self._components: dict = {}  # name → instance

    def register(self, *components) -> None:
        for component in components:
            cls = type(component)
            _collect_dependencies(cls, self._components)
            self._components[cls.__name__] = component

    def start(self) -> None:
        cache = {}
        ordered = sorted(
            self._components.values(),
            key=lambda c: _resolve_index(type(c), cache),
        )
        for component in ordered:
            component.start()
            if hasattr(component, "ready_event"):
                component.ready_event.wait()

    def stop(self) -> None:
        cache = {}
        ordered = sorted(
            self._components.values(),
            key=lambda c: _resolve_index(type(c), cache),
            reverse=True,
        )
        for component in ordered:
            component.stop()
