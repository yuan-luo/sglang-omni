1.

class _SGLangAttr:
    """Lazy proxy for attributes provided by the sglang_ar module."""

    def __init__(self, name: str):
        self._name = name
        self._value = None

    def _resolve(self):
        if self._value is None:
            self._value = globals()["__getattr__"](self._name)
        return self._value

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)


# Provide module-level bindings for lazily loaded SGLang exports.
SGLangARRequestData = _SGLangAttr("SGLangARRequestData")
SGLangBatchPlanner = _SGLangAttr("SGLangBatchPlanner")
SGLangResourceManager = _SGLangAttr("SGLangResourceManager")
SGLangOutputProcessor = _SGLangAttr("SGLangOutputProcessor")
SGLangIterationController = _SGLangAttr("SGLangIterationController")
SGLangModelRunner = _SGLangAttr("SGLangModelRunner")


I strongly do not like this code style, I think this class can be removed. It's too nasty to have it, with those method like __getattr__, __call__

2.

def _get_inner_text_model(model: Any) -> Any | None:
    """Navigate model wrappers to find the inner text model with layers + embed_tokens.

    Handles various nesting patterns:
      - model.layers                       (direct)
      - model.model.layers                 (e.g., LlamaForCausalLM)
      - model.model.model.layers           (deeper wrappers)
      - model.thinker.model.layers         (Qwen3OmniMoeForConditionalGeneration)
    """
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
        # Qwen3-Omni: model.thinker.model has layers + embed_tokens
        getattr(getattr(model, "thinker", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if getattr(candidate, "layers", None) is not None:
            return candidate
    return None

def _get_model_layers(model: Any) -> Any | None:
    """Navigate model wrapper to find the decoder layers list."""
    inner = _get_inner_text_model(model)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None and hasattr(layers, "__len__"):
            return layers
    return None

def _get_embed_tokens(model: Any) -> Any | None:
    """Navigate model wrapper to find the embed_tokens module."""
    inner = _get_inner_text_model(model)
    if inner is not None:
        return getattr(inner, "embed_tokens", None)
    return None

These three functions are really ugly. I hate getattr, and I think this function is over protecting. We can write it rather clean.
