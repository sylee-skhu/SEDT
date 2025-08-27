# models/__init__.py
"""
모델 팩토리 & 레지스트리
- 현재는 SEDT만 등록되어 있음
- 향후 다른 모델은 register_model 데코레이터로 손쉽게 추가 가능
    예)
        from . import register_model
        from .foo import FOO

        @register_model("FOO")
        def build_foo(args):
            return FOO(num_classes=args.NUM_CLASSES, ...)

사용:
    from models import create_model
    model = create_model(args)
"""

from typing import Callable, Dict, Any
import inspect

# -------------------------
# 레지스트리 기본구현
# -------------------------
MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_model(name: str):
    """모델 빌더 함수를 등록하는 데코레이터"""
    def _wrap(fn: Callable[..., Any]):
        MODEL_REGISTRY[name.upper()] = fn
        return fn
    return _wrap

def list_models():
    return sorted(MODEL_REGISTRY.keys())

def _filter_kwargs(callable_obj: Callable[..., Any], kw: Dict[str, Any]) -> Dict[str, Any]:
    """callable 시그니처에 존재하는 키만 남겨서 안전하게 전달 (키 대소문자 무시)"""
    if not kw:
        return {}
    sig = inspect.signature(callable_obj)
    params_lc = {p.lower() for p in sig.parameters.keys()}
    out = {}
    for k, v in kw.items():
        kk = k.lower() if isinstance(k, str) else k
        if kk in params_lc:
            out[kk] = v
    return out

# -------------------------
# SEDT 등록
# -------------------------
from .sedt import SEDT  # 현재 패키지 내 유일한 모델

@register_model("SEDT")
def build_sedt(args):
    """
    args로부터 SEDT를 구성하여 반환.
    필수:
        - args.MODEL_NAME == 'SEDT'
    선택:
        - args.ARCH in {'SEDT_T','SEDT_S','SEDT_B'}
        - args.MODEL_KWARGS 또는 args.MODEL_EXTRAS: dict
    """
    arch = getattr(args, "ARCH", "SEDT_S").upper()

    # 아키텍처 프리셋
    if arch == "SEDT_T":
        kwargs = dict(embed_dim=16)
    elif arch == "SEDT_S":
        kwargs = dict(embed_dim=32)
    elif arch == "SEDT_B":
        kwargs = dict(embed_dim=32, depths=[1, 2, 8, 8, 2, 8, 8, 2, 1])
    else:
        raise NotImplementedError(f"Unknown ARCH for SEDT: {arch}")

    extras = getattr(args, "MODEL_KWARGS", None) or getattr(args, "MODEL_EXTRAS", None)
    if isinstance(extras, dict):
        kwargs.update(_filter_kwargs(SEDT.__init__, extras))

    return SEDT(**kwargs)

# -------------------------
# 공용 팩토리
# -------------------------
def create_model(args):
    """
    args.MODEL_NAME 에 해당하는 모델을 생성.
    """
    name = getattr(args, "MODEL_NAME", None)
    if not name:
        raise ValueError("CONFIG: MODEL_NAME must be specified.")
    name = name.upper()

    if name not in MODEL_REGISTRY:
        available = ", ".join(list_models())
        raise NotImplementedError(f"Unknown model: {name}. Available: {available}")

    builder = MODEL_REGISTRY[name]
    return builder(args)
