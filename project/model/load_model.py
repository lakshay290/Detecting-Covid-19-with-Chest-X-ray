from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Callable


def _resolve_load_model() -> Callable[[str | Path], Any] | None:
    # Try TensorFlow Keras first, then standalone Keras.
    for module_name in ("tensorflow.keras.models", "keras.models"):
        try:
            module = import_module(module_name)
        except Exception:
            continue

        load_fn = getattr(module, "load_model", None)
        if callable(load_fn):
            return load_fn

    return None


@lru_cache(maxsize=1)
def get_model(model_path: str | Path) -> Any | None:
    path = Path(model_path)
    if not path.exists():
        return None

    load_model = _resolve_load_model()
    if load_model is None:
        return None

    try:
        return load_model(path)
    except Exception:
        return None
