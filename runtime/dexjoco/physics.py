from contextlib import AbstractContextManager
from typing import Any
class CaptureMjStep(AbstractContextManager["CaptureMjStep"]):
    """Temporarily observe each MuJoCo physics substep used by ``env.step``."""

    def __init__(self, collector: Any, model: Any, data: Any) -> None:
        self.collector, self.model, self.data = collector, model, data
        self.mujoco = collector.mujoco
        self.original = None

    def __enter__(self) -> "CaptureMjStep":
        self.original = self.mujoco.mj_step

        def observed_step(model: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            result = self.original(model, data, *args, **kwargs)
            if model is self.model and data is self.data:
                self.collector.capture_substep(data)
            return result

        self.mujoco.mj_step = observed_step
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.original is not None:
            self.mujoco.mj_step = self.original
