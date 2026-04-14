from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch


@dataclass
class EncoderBundle:
    name: str
    model: Any
    preprocess: Optional[Callable]
    device: torch.device
    supports_text: bool
    supports_slide: bool
    extra: Optional[dict] = None


def _default_models_root() -> Path:
    return Path(__file__).resolve().parents[3] / "models"


def _pick_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_titan_import_path(models_root: Path) -> None:
    root_str = str(models_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_conch(models_root: Path, device: torch.device, hf_token: Optional[str]) -> EncoderBundle:
    from conch.open_clip_custom import create_model_from_pretrained

    checkpoint = models_root / "conch" / "pytorch_model.bin"
    if not checkpoint.exists():
        raise FileNotFoundError(f"CONCH checkpoint not found: {checkpoint}")

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        str(checkpoint),
        hf_auth_token=hf_token,
    )
    model = model.to(device).eval()
    return EncoderBundle(
        name="conch",
        model=model,
        preprocess=preprocess,
        device=device,
        supports_text=True,
        supports_slide=False,
        extra={"checkpoint": str(checkpoint)},
    )


def _build_conchv15_local(c15_mod: Any, conch_cfg: Any, checkpoint_path: Path) -> tuple[Any, Callable]:
    model = c15_mod.VisionTransformer(
        patch_size=conch_cfg.patch_size,
        embed_dim=conch_cfg.context_dim,
        depth=conch_cfg.depth,
        num_heads=conch_cfg.num_heads,
        mlp_ratio=conch_cfg.mlp_ratio,
        qkv_bias=conch_cfg.qkv_bias,
        init_values=conch_cfg.init_values,
    )
    attn_pooler = c15_mod.AttentionalPooler(
        d_model=conch_cfg.embed_dim,
        context_dim=conch_cfg.context_dim,
        n_queries=conch_cfg.pooler_n_queries_contrast,
    )
    model = c15_mod.EncoderWithAttentionalPooler(
        encoder=model,
        attn_pooler_contrast=attn_pooler,
        embed_dim=conch_cfg.embed_dim,
    )

    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    img_size = 448
    eval_transform = c15_mod.T.Compose(
        [
            c15_mod.T.Resize(img_size, interpolation=c15_mod.T.InterpolationMode.BILINEAR),
            c15_mod.T.CenterCrop(img_size),
            c15_mod.T.ToTensor(),
            c15_mod.T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return model, eval_transform


def _load_titan_local(titan_dir: Path) -> Any:
    _ensure_titan_import_path(titan_dir.parent)
    mt = importlib.import_module("TITAN.modeling_titan")
    ct = importlib.import_module("TITAN.configuration_titan")

    config_path = titan_dir / "config.json"
    weights_path = titan_dir / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"TITAN config missing: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"TITAN weights missing: {weights_path}")

    cfg_dict = json.loads(config_path.read_text())
    titan_cfg = ct.TitanConfig(
        vision_config=cfg_dict.get("vision_config", {}),
        text_config=cfg_dict.get("text_config", {}),
        conch_config=cfg_dict.get("conch_config", {}),
    )
    from transformers import PreTrainedTokenizerFast

    original_from_pretrained = PreTrainedTokenizerFast.from_pretrained
    local_titan_dir = str(titan_dir)

    def _from_pretrained_local(name_or_path: str, *args: Any, **kwargs: Any):
        if name_or_path == "MahmoodLab/TITAN":
            return original_from_pretrained(local_titan_dir, *args, **kwargs)
        return original_from_pretrained(name_or_path, *args, **kwargs)

    PreTrainedTokenizerFast.from_pretrained = _from_pretrained_local
    try:
        titan = mt.Titan(titan_cfg)
    finally:
        PreTrainedTokenizerFast.from_pretrained = original_from_pretrained

    from safetensors.torch import load_file

    state_dict = load_file(str(weights_path), device="cpu")
    missing, unexpected = titan.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"TITAN missing keys during load: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"TITAN unexpected keys during load: {unexpected[:8]}")

    return titan


def _load_conchv15_from_titan(models_root: Path, device: torch.device) -> EncoderBundle:
    titan_dir = models_root / "TITAN"
    if not titan_dir.exists():
        raise FileNotFoundError(f"TITAN folder not found: {titan_dir}")

    _ensure_titan_import_path(models_root)
    c15_mod = importlib.import_module("TITAN.conch_v1_5")

    config_path = titan_dir / "config.json"
    conch_ckpt = titan_dir / "conch_v1_5_pytorch_model.bin"
    if not config_path.exists():
        raise FileNotFoundError(f"TITAN config missing: {config_path}")
    if not conch_ckpt.exists():
        raise FileNotFoundError(f"CONCH v1.5 checkpoint missing: {conch_ckpt}")

    cfg_dict = json.loads(config_path.read_text())
    ct = importlib.import_module("TITAN.configuration_titan")
    conch_cfg = ct.ConchConfig(**cfg_dict.get("conch_config", {}))

    conch_v15, preprocess = _build_conchv15_local(c15_mod, conch_cfg, conch_ckpt)
    conch_v15 = conch_v15.to(device).eval()

    return EncoderBundle(
        name="conchv1_5",
        model=conch_v15,
        preprocess=preprocess,
        device=device,
        supports_text=False,
        supports_slide=False,
        extra={
            "source": "local TITAN.conch_v1_5",
            "checkpoint": str(conch_ckpt),
        },
    )


def _load_titan(models_root: Path, device: torch.device) -> EncoderBundle:
    titan_dir = models_root / "TITAN"
    if not titan_dir.exists():
        raise FileNotFoundError(f"TITAN folder not found: {titan_dir}")

    titan = _load_titan_local(titan_dir).to(device).eval()

    _ensure_titan_import_path(models_root)
    c15_mod = importlib.import_module("TITAN.conch_v1_5")
    cfg_dict = json.loads((titan_dir / "config.json").read_text())
    ct = importlib.import_module("TITAN.configuration_titan")
    conch_cfg = ct.ConchConfig(**cfg_dict.get("conch_config", {}))
    patch_encoder, preprocess = _build_conchv15_local(
        c15_mod,
        conch_cfg,
        titan_dir / "conch_v1_5_pytorch_model.bin",
    )
    patch_encoder = patch_encoder.to(device).eval()

    return EncoderBundle(
        name="titan",
        model=titan,
        preprocess=preprocess,
        device=device,
        supports_text=True,
        supports_slide=True,
        extra={
            "patch_encoder": patch_encoder,
            "titan_dir": str(titan_dir),
            "usage": "Use model.encode_slide_from_patch_features(features, coords, patch_size_lv0)",
        },
    )


def load_encoder(
    encoder: str,
    models_root: Optional[str] = None,
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> EncoderBundle:
    """
    Unified loader for pathology encoders.

    Supported encoders:
    - conch: CONCH image-text model (ViT-B/16)
    - conchv1_5: CONCH v1.5 patch encoder (loaded from local TITAN assets)
    - titan: TITAN slide-level model (+ patch encoder in bundle.extra['patch_encoder'])
    """
    normalized = encoder.strip().lower()
    root = Path(models_root).expanduser().resolve() if models_root else _default_models_root()
    dev = _pick_device(device)

    if normalized == "conch":
        return _load_conch(root, dev, hf_token)
    if normalized in {"conchv1_5", "conch_v1_5", "conch15"}:
        return _load_conchv15_from_titan(root, dev)
    if normalized == "titan":
        return _load_titan(root, dev)

    raise ValueError(f"Unsupported encoder '{encoder}'. Choose from: conch, conchv1_5, titan")
