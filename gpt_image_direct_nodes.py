"""
ComfyUI-GPT-Image-Direct
========================
Direct OpenAI Images API node for ComfyUI. Bypasses the Comfy.org credit
system (and its markup) by calling https://api.openai.com/v1/images/* with
your own API key.

  - GPT Image 2.6 and 2.5 Flare / Sunburst, GPT Image 2, 1.5 and 1-mini
  - Text-to-image and reference-image editing (up to 16 refs) with optional mask
  - Exact USD cost per run computed from the API's own `usage` token counts
  - Zero extra dependencies: uses `requests` + Pillow, both shipped with ComfyUI

Key resolution order: node input > OPENAI_API_KEY env var > openai_api_key.txt
in the ComfyUI root.
"""

import base64
import hashlib
import io
import json
import os
import time

import numpy as np
import torch
from PIL import Image

try:
    import requests
except ImportError:  # pragma: no cover - requests ships with ComfyUI
    requests = None

try:
    import folder_paths
except ImportError:  # running outside ComfyUI (tests)
    folder_paths = None


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CATEGORY = "AKURATE/GPT Image Direct"
LOG = "[GPT Image Direct]"

DEFAULT_API_BASE = "https://api.openai.com/v1"

MODELS = [
    "gpt-image-2.6-flare",
    "gpt-image-2.6-sunburst",
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1-mini",
]

# USD per 1M tokens, from developers.openai.com/api/docs/pricing (2.6 rows checked 2026-11-08).
# Keys: text_in, cached_in, image_in, image_out.
PRICE_PER_1M = {
    "gpt-image-2.6-flare":    {"text_in": 5.00, "cached_in": 1.25, "image_in": 8.00, "image_out": 30.00},
    "gpt-image-2.6-sunburst": {"text_in": 5.00, "cached_in": 1.25, "image_in": 8.00, "image_out": 30.00},
    "gpt-image-2.5-flare":    {"text_in": 5.00, "cached_in": 1.25, "image_in": 8.00, "image_out": 30.00},
    "gpt-image-2.5-sunburst": {"text_in": 5.00, "cached_in": 1.25, "image_in": 8.00, "image_out": 30.00},
    "gpt-image-2":            {"text_in": 5.00, "cached_in": 1.25, "image_in": 8.00, "image_out": 30.00},
    "gpt-image-1.5":          {"text_in": 5.00, "cached_in": 1.25, "image_in": 8.00, "image_out": 32.00},
    "gpt-image-1-mini":       {"text_in": 2.00, "cached_in": 0.20, "image_in": 2.50, "image_out": 8.00},
}

# Rough pre-flight estimate of OUTPUT tokens per quality at ~1MP so the console
# shows an expected cost before the call. The post-call figure is exact.
EST_OUTPUT_TOKENS = {"low": 196, "medium": 620, "high": 1756, "xhigh": 3000, "max": 7024, "auto": 620}

QUALITIES_25 = ["auto", "low", "medium", "high", "xhigh", "max"]
QUALITIES_LEGACY = ["auto", "low", "medium", "high"]

SIZES_2X = [
    "auto",
    "1024x1024", "1024x1536", "1536x1024",
    "2048x2048", "2048x1152", "1152x2048",
    "3840x2160", "2160x3840",
    "custom",
]
SIZES_LEGACY = ["auto", "1024x1024", "1024x1536", "1536x1024"]

MODEL_SIZES = {
    "gpt-image-2.6-flare": SIZES_2X,
    "gpt-image-2.6-sunburst": SIZES_2X,
    "gpt-image-2.5-flare": SIZES_2X,
    "gpt-image-2.5-sunburst": SIZES_2X,
    "gpt-image-2": SIZES_2X,
    "gpt-image-1.5": SIZES_LEGACY,
    "gpt-image-1-mini": SIZES_LEGACY,
}
MODEL_QUALITIES = {
    "gpt-image-2.6-flare": QUALITIES_25,
    "gpt-image-2.6-sunburst": QUALITIES_25,
    "gpt-image-2.5-flare": QUALITIES_25,
    "gpt-image-2.5-sunburst": QUALITIES_25,
    "gpt-image-2": QUALITIES_LEGACY,
    "gpt-image-1.5": QUALITIES_LEGACY,
    "gpt-image-1-mini": QUALITIES_LEGACY,
}

# Custom-size constraints for the 2.x line (mirrors OpenAI's documented limits).
CUSTOM_MIN_EDGE = 480
CUSTOM_MAX_EDGE = 3840
CUSTOM_MIN_PIXELS = 655_360
CUSTOM_MAX_PIXELS = 8_294_400
CUSTOM_MAX_RATIO = 3.0

REFERENCE_MAX_SIZE = ["2048", "1536", "1024", "original"]
MAX_REFERENCE_IMAGES = 16


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def _comfy_root():
    if folder_paths:
        return os.path.dirname(folder_paths.get_output_directory())
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_api_key(api_key_input=""):
    """Resolve API key: direct input > env var > file in ComfyUI root."""
    if api_key_input and api_key_input.strip():
        return api_key_input.strip()

    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    key_file = os.path.join(_comfy_root(), "openai_api_key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as f:
            file_key = f.read().strip()
        if file_key:
            return file_key

    raise ValueError(
        f"{LOG} No API key found. Provide one via:\n"
        "  1. The api_key input on the node\n"
        "  2. OPENAI_API_KEY environment variable\n"
        "  3. An openai_api_key.txt file in your ComfyUI root directory\n\n"
        "Get a key at https://platform.openai.com/api-keys\n"
        "GPT Image models also need one-time Organization Verification at\n"
        "https://platform.openai.com/settings/organization/general"
    )


def _tensor_to_pil(tensor):
    """Single image tensor (H,W,C float 0-1) -> PIL RGB."""
    arr = (tensor.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, "RGBA").convert("RGB")
    return Image.fromarray(arr, "RGB")


def _pil_to_tensors(pil_img):
    """PIL -> (IMAGE tensor (1,H,W,3), MASK tensor (1,H,W)) following LoadImage
    conventions: mask = 1 - alpha (1 where transparent)."""
    if pil_img.mode == "RGBA":
        alpha = np.array(pil_img.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - alpha
    else:
        mask = np.zeros((pil_img.height, pil_img.width), dtype=np.float32)
    rgb = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


def _downscale(pil_img, max_edge):
    """Fit inside max_edge x max_edge (keeps aspect). None = untouched."""
    if max_edge is None:
        return pil_img
    w, h = pil_img.size
    if max(w, h) <= max_edge:
        return pil_img
    scale = max_edge / float(max(w, h))
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return pil_img.resize(new_size, Image.LANCZOS)


def _flatten_references(inputs):
    """image_1..4 sockets + an IMAGE batch -> list of single (H,W,C) tensors, in order."""
    flat = []
    for t in inputs:
        if t is None:
            continue
        if t.dim() == 3:
            flat.append(t)
        else:
            flat.extend(t[i] for i in range(t.shape[0]))
    return flat


def _png_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _get_output_dir():
    if folder_paths:
        return folder_paths.get_output_directory()
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _save_image(pil_img, ext, index):
    out_dir = os.path.join(_get_output_dir(), "gpt_image_direct")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"gpt_image_{stamp}_{index:02d}.{ext}"
    filepath = os.path.join(out_dir, filename)
    save_img = pil_img
    if ext == "jpg" and pil_img.mode == "RGBA":
        save_img = pil_img.convert("RGB")
    save_img.save(filepath)
    return filepath


def _parse_size(size_str):
    w, h = size_str.lower().split("x")
    return int(w), int(h)


def _validate_custom_size(width, height):
    if width % 16 or height % 16:
        raise ValueError(f"{LOG} custom size must be multiples of 16, got {width}x{height}")
    if not (CUSTOM_MIN_EDGE <= width <= CUSTOM_MAX_EDGE and CUSTOM_MIN_EDGE <= height <= CUSTOM_MAX_EDGE):
        raise ValueError(
            f"{LOG} custom edges must be within {CUSTOM_MIN_EDGE}-{CUSTOM_MAX_EDGE} px, got {width}x{height}"
        )
    ratio = max(width, height) / float(min(width, height))
    if ratio > CUSTOM_MAX_RATIO:
        raise ValueError(f"{LOG} custom aspect ratio must be <= 3:1, got {width}x{height}")
    pixels = width * height
    if not (CUSTOM_MIN_PIXELS <= pixels <= CUSTOM_MAX_PIXELS):
        raise ValueError(
            f"{LOG} custom total pixels must be {CUSTOM_MIN_PIXELS:,}-{CUSTOM_MAX_PIXELS:,}, got {pixels:,}"
        )


def _resolve_size(model, size, custom_width, custom_height):
    """Return the `size` string to send, or None for 'auto'. Falls back for legacy models."""
    if size == "auto":
        return None
    if size == "custom":
        if "custom" not in MODEL_SIZES.get(model, SIZES_LEGACY):
            print(f"{LOG} {model} has no custom sizes, falling back to 1024x1024")
            return "1024x1024"
        _validate_custom_size(custom_width, custom_height)
        return f"{custom_width}x{custom_height}"
    if size not in MODEL_SIZES.get(model, SIZES_LEGACY):
        print(f"{LOG} {model} doesn't support {size}, falling back to 1024x1024")
        return "1024x1024"
    return size


def _resolve_quality(model, quality):
    allowed = MODEL_QUALITIES.get(model, QUALITIES_LEGACY)
    if quality in allowed:
        return quality
    print(f"{LOG} {model} doesn't support quality={quality}, falling back to high")
    return "high"


def compute_cost(model, usage):
    """Exact USD cost from the API's `usage` block. Returns (usd, breakdown dict)."""
    rates = PRICE_PER_1M.get(model, PRICE_PER_1M["gpt-image-2"])
    usage = usage or {}
    details = usage.get("input_tokens_details") or {}
    text_in = int(details.get("text_tokens", 0) or 0)
    image_in = int(details.get("image_tokens", 0) or 0)
    cached_in = int(details.get("cached_tokens", 0) or 0)
    if not details:
        text_in = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)

    # Cached tokens are billed at the cached rate; OpenAI reports them as part
    # of the input totals, so subtract them from the uncached buckets.
    cached_remaining = cached_in
    image_uncached = max(0, image_in - cached_remaining)
    cached_remaining = max(0, cached_remaining - image_in)
    text_uncached = max(0, text_in - cached_remaining)

    usd = (
        text_uncached * rates["text_in"]
        + image_uncached * rates["image_in"]
        + cached_in * rates["cached_in"]
        + out * rates["image_out"]
    ) / 1_000_000.0
    return usd, {"text_in": text_in, "image_in": image_in, "cached_in": cached_in, "out": out}


def estimate_cost(model, quality, n, n_refs):
    """Pre-flight rough estimate (output tokens at ~1MP; refs ~1000 tokens each)."""
    rates = PRICE_PER_1M.get(model, PRICE_PER_1M["gpt-image-2"])
    out_tokens = EST_OUTPUT_TOKENS.get(quality, 620) * n
    in_tokens = 1000 * n_refs + 50
    return (out_tokens * rates["image_out"] + in_tokens * rates["image_in"]) / 1_000_000.0


def _friendly_http_error(status, body_text):
    """Turn an OpenAI error payload into a readable ComfyUI error."""
    message = body_text
    code = ""
    try:
        payload = json.loads(body_text)
        err = payload.get("error", {})
        message = err.get("message", body_text)
        code = err.get("code") or err.get("type") or ""
    except (ValueError, AttributeError):
        pass

    hint = ""
    if status in (401, 403):
        hint = (
            "\nCheck the key, and make sure your organization is verified for GPT Image models:\n"
            "https://platform.openai.com/settings/organization/general"
        )
    elif status == 429:
        hint = "\nRate limit or quota hit. Check https://platform.openai.com/settings/organization/limits"
    elif status == 400 and "moderation" in message.lower():
        hint = "\nThe prompt or a reference image was blocked by OpenAI's safety system."
    return f"{LOG} OpenAI API error {status}{(' ' + code) if code else ''}: {message}{hint}"


def _post(url, api_key, timeout, json_body=None, data=None, files=None):
    if requests is None:
        raise ImportError(f"{LOG} the `requests` package is missing. Run: pip install requests")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.post(url, headers=headers, json=json_body, data=data, files=files, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"{LOG} request timed out after {timeout}s. Raise timeout_sec for max quality / 4K.")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"{LOG} could not reach {url}: {e}")
    if resp.status_code >= 400:
        raise RuntimeError(_friendly_http_error(resp.status_code, resp.text))
    try:
        return resp.json()
    except ValueError:
        raise RuntimeError(f"{LOG} non-JSON response from API: {resp.text[:400]}")


def build_generation_payload(model, prompt, quality, size, background, output_format,
                             output_compression, moderation, n):
    payload = {
        "model": model,
        "prompt": prompt,
        "n": int(n),
        "quality": quality,
        "moderation": moderation,
        "output_format": output_format,
    }
    if size:
        payload["size"] = size
    if background != "auto":
        payload["background"] = background
    if output_format in ("jpeg", "webp"):
        payload["output_compression"] = int(output_compression)
    return payload


def build_edit_files(ref_pils, mask_tensor):
    """Multipart file tuples for /images/edits. Mask (B,H,W float, 1 = edit) becomes
    an RGBA PNG whose alpha is 0 where the model should repaint."""
    files = []
    field = "image" if len(ref_pils) == 1 else "image[]"
    for i, pil in enumerate(ref_pils):
        files.append((field, (f"image_{i}.png", _png_bytes(pil), "image/png")))

    if mask_tensor is not None:
        if len(ref_pils) != 1:
            raise ValueError(f"{LOG} a mask requires exactly one reference image, got {len(ref_pils)}")
        ref = ref_pils[0]
        m = mask_tensor[0] if mask_tensor.dim() == 3 else mask_tensor
        m_np = (m.detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
        m_pil = Image.fromarray(m_np, "L")
        if m_pil.size != ref.size:
            m_pil = m_pil.resize(ref.size, Image.NEAREST)
        alpha = Image.fromarray(255 - np.array(m_pil), "L")
        rgba = Image.new("RGBA", ref.size, (0, 0, 0, 255))
        rgba.putalpha(alpha)
        files.append(("mask", ("mask.png", _png_bytes(rgba), "image/png")))
    return files


# ============================================================================
# NODE: GPT IMAGE GENERATE (Direct API)
# ============================================================================

class GPTImageGenerate:
    """
    Generate or edit images with OpenAI GPT Image models using your own API key.
    Attach reference images (IMAGE batch) to switch to the edits endpoint; add a
    MASK (white = repaint) with a single reference for inpainting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "What to generate, or how to edit the reference images."}),
                "model": (MODELS, {"default": "gpt-image-2.6-flare",
                                   "tooltip": "Flare: fastest, everyday quality. Sunburst: slower, more precise (text, layouts, heavy edits). Same price per token."}),
                "quality": (QUALITIES_25, {"default": "low",
                                           "tooltip": "Biggest cost lever. low ~$0.006, medium ~$0.02, high ~$0.05, max ~$0.21 per 1MP image. xhigh/max are 2.5/2.6 only."}),
                "size": (SIZES_2X, {"default": "1024x1024", "tooltip": "auto lets the model pick. 2048+/4K/custom are 2.x only."}),
                "custom_width": ("INT", {"default": 1024, "min": CUSTOM_MIN_EDGE, "max": CUSTOM_MAX_EDGE, "step": 16, "tooltip": "Used when size = custom. Multiple of 16, aspect <= 3:1."}),
                "custom_height": ("INT", {"default": 1024, "min": CUSTOM_MIN_EDGE, "max": CUSTOM_MAX_EDGE, "step": 16, "tooltip": "Used when size = custom."}),
                "background": (["auto", "opaque", "transparent"], {"default": "auto", "tooltip": "transparent needs png or webp; alpha comes out on the mask output."}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png", "tooltip": "File format returned and saved. Does not change cost."}),
                "output_compression": ("INT", {"default": 90, "min": 0, "max": 100, "step": 1, "tooltip": "jpeg/webp only."}),
                "moderation": (["auto", "low"], {"default": "low", "tooltip": "Content filter strictness."}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "Images per run. Cost scales linearly."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True,
                                 "tooltip": "Not sent to OpenAI (no seed support). Change it to force a re-run and a new cache_key."}),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Reference image (any size). Any connected reference switches to the edits endpoint."}),
                "image_2": ("IMAGE", {"tooltip": "Reference image."}),
                "image_3": ("IMAGE", {"tooltip": "Reference image."}),
                "image_4": ("IMAGE", {"tooltip": "Reference image."}),
                "images": ("IMAGE", {"tooltip": "Reference image batch, appended after image_1..4. 16 references total."}),
                "mask": ("MASK", {"tooltip": "Inpaint mask, white = repaint. Needs exactly one reference image in total."}),
                "reference_max_size": (REFERENCE_MAX_SIZE, {"default": "2048", "tooltip": "Downscale refs before upload. Smaller = fewer input tokens = cheaper."}),
                "input_fidelity": (["auto", "low", "high"], {"default": "auto", "tooltip": "Edits only. high preserves reference detail but costs more input tokens. auto = don't send."}),
                "api_key": ("STRING", {"default": "", "tooltip": "Overrides OPENAI_API_KEY env var and openai_api_key.txt in the ComfyUI root."}),
                "api_base": ("STRING", {"default": DEFAULT_API_BASE, "tooltip": "OpenAI-compatible base URL (Azure/fal/proxy)."}),
                "timeout_sec": ("INT", {"default": 300, "min": 30, "max": 1800, "step": 10, "tooltip": "max quality at 4K on Sunburst can take minutes."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("images", "mask", "cost_info", "cache_key")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, prompt, model, quality, size, custom_width, custom_height, background,
                 output_format, output_compression, moderation, n, seed,
                 image_1=None, image_2=None, image_3=None, image_4=None,
                 images=None, mask=None, reference_max_size="2048", input_fidelity="auto",
                 api_key="", api_base=DEFAULT_API_BASE, timeout_sec=300):

        if not prompt or not prompt.strip():
            raise ValueError(f"{LOG} prompt is empty")
        prompt = prompt.strip()

        resolved_key = _resolve_api_key(api_key)
        base = (api_base or DEFAULT_API_BASE).strip().rstrip("/")

        quality = _resolve_quality(model, quality)
        size_str = _resolve_size(model, size, custom_width, custom_height)

        if background == "transparent" and output_format == "jpeg":
            print(f"{LOG} transparent background needs png/webp, switching output_format to png")
            output_format = "png"

        # --- reference images ---------------------------------------------------
        ref_pils = []
        ref_hash = ""
        ref_tensors = _flatten_references([image_1, image_2, image_3, image_4, images])
        if ref_tensors:
            max_edge = None if reference_max_size == "original" else int(reference_max_size)
            if len(ref_tensors) > MAX_REFERENCE_IMAGES:
                raise ValueError(f"{LOG} at most {MAX_REFERENCE_IMAGES} reference images, got {len(ref_tensors)}")
            hasher = hashlib.sha1()
            for t in ref_tensors:
                pil = _downscale(_tensor_to_pil(t), max_edge)
                ref_pils.append(pil)
                hasher.update(pil.tobytes()[::97])
            if mask is not None:
                hasher.update(mask.detach().cpu().numpy().tobytes()[::97])
            ref_hash = hasher.hexdigest()[:12]
            print(f"{LOG} {len(ref_pils)} reference image(s) attached"
                  f"{' + mask' if mask is not None else ''} (max edge {reference_max_size})")
        elif mask is not None:
            raise ValueError(f"{LOG} a mask needs a reference image on the images input")

        cache_key = "|".join([
            model, quality, size_str or "auto", background, output_format,
            str(n), str(seed), ref_hash, prompt,
        ])

        est = estimate_cost(model, quality, n, len(ref_pils))
        print(f"{LOG} {model} | quality={quality} | size={size_str or 'auto'} | n={n} | est. ~${est:.4f}")

        # --- call -----------------------------------------------------------------
        start = time.time()
        if ref_pils:
            data = build_generation_payload(model, prompt, quality, size_str, background,
                                            output_format, output_compression, moderation, n)
            if input_fidelity != "auto":
                data["input_fidelity"] = input_fidelity
            files = build_edit_files(ref_pils, mask)
            result = _post(f"{base}/images/edits", resolved_key, timeout_sec, data=data, files=files)
        else:
            payload = build_generation_payload(model, prompt, quality, size_str, background,
                                               output_format, output_compression, moderation, n)
            result = _post(f"{base}/images/generations", resolved_key, timeout_sec, json_body=payload)
        elapsed = time.time() - start

        # --- decode -------------------------------------------------------------
        items = result.get("data") or []
        if not items:
            raise RuntimeError(f"{LOG} API returned no images: {json.dumps(result)[:400]}")

        ext = {"png": "png", "jpeg": "jpg", "webp": "webp"}[output_format]
        image_tensors, mask_tensors, saved = [], [], []
        for idx, item in enumerate(items):
            b64 = item.get("b64_json")
            if not b64:
                raise RuntimeError(f"{LOG} image {idx} has no b64_json payload: {json.dumps(item)[:200]}")
            pil = Image.open(io.BytesIO(base64.b64decode(b64)))
            pil.load()
            if pil.mode not in ("RGB", "RGBA"):
                pil = pil.convert("RGBA" if "A" in pil.getbands() else "RGB")
            saved.append(os.path.basename(_save_image(pil, ext, idx)))
            img_t, mask_t = _pil_to_tensors(pil)
            image_tensors.append(img_t)
            mask_tensors.append(mask_t)

        shapes = {t.shape[1:3] for t in image_tensors}
        if len(shapes) > 1:
            # Different sizes can't share a batch; resize to the first.
            h, w = image_tensors[0].shape[1:3]
            image_tensors = [torch.nn.functional.interpolate(t.permute(0, 3, 1, 2), size=(h, w), mode="bilinear").permute(0, 2, 3, 1)
                             for t in image_tensors]
            mask_tensors = [torch.nn.functional.interpolate(t.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)
                            for t in mask_tensors]
        out_images = torch.cat(image_tensors, dim=0)
        out_masks = torch.cat(mask_tensors, dim=0)

        # --- cost ----------------------------------------------------------------
        usage = result.get("usage") or {}
        usd, tokens = compute_cost(model, usage)
        cost_source = "exact" if usage else "estimated"
        if not usage:
            usd = est
        w, h = out_images.shape[2], out_images.shape[1]
        cost_info = (
            f"${usd:.4f} USD ({cost_source}) | {model} {quality} {w}x{h} x{len(items)} | {elapsed:.1f}s | "
            f"tokens: {tokens['text_in']} text-in, {tokens['image_in']} img-in, "
            f"{tokens['cached_in']} cached, {tokens['out']} out | {', '.join(saved)}"
        )
        print(f"{LOG} Done. {cost_info}")

        return (out_images, out_masks, cost_info, cache_key)


# ============================================================================
# MAPPINGS
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "GPTImageGenerate": GPTImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageGenerate": "GPT Image Generate (Direct API)",
}
