"""
Offline test for ComfyUI-GPT-Image-Direct.

Mocks `requests.post` so no money is spent, then checks:
  - generation payload shape (JSON path) and edit multipart shape (files path)
  - mask -> RGBA alpha inversion and auto-resize to the reference
  - cost math from a real-shaped `usage` block (with a negative control)
  - quality/size fallbacks for legacy models, custom-size validation
  - error surfacing (401 -> verification hint)
  - IMAGE/MASK tensor shapes and dtype, transparent alpha -> mask

Run with the embedded python:
  F:/.../python_embeded/python.exe tools/test_gpt_image_direct.py

Pass --live to additionally fire ONE real low-quality 1024x1024 Flare call
(needs openai_api_key.txt in the ComfyUI root or OPENAI_API_KEY). ~$0.006.
"""

import base64
import io
import json
import os
import sys
import types

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)

import gpt_image_direct_nodes as gid  # noqa: E402

FAILS = []
PASSES = 0


def check(cond, name):
    global PASSES
    if cond:
        PASSES += 1
        print(f"  PASS {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL {name}")


def _b64_png(w, h, rgba=False):
    mode = "RGBA" if rgba else "RGB"
    img = Image.new(mode, (w, h), (200, 100, 50, 128) if rgba else (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


CAPTURED = []


def make_fake_post(status=200, n=1, w=64, h=64, rgba=False, usage=None):
    def fake_post(url, headers=None, json=None, data=None, files=None, timeout=None):
        CAPTURED.append({"url": url, "headers": headers, "json": json, "data": data, "files": files, "timeout": timeout})
        if status != 200:
            return FakeResponse(status, {"error": {"message": "Your organization must be verified to use the model `gpt-image-2.5-flare`.", "type": "invalid_request_error", "code": "org_verification_required"}})
        payload = {"created": 1, "data": [{"b64_json": _b64_png(w, h, rgba)} for _ in range(n)]}
        if usage is not None:
            payload["usage"] = usage
        return FakeResponse(200, payload)
    return fake_post


def run_offline():
    os.environ["OPENAI_API_KEY"] = "sk-test-not-real"
    gid.folder_paths = None  # keep mock outputs out of the real ComfyUI output folder
    node = gid.GPTImageGenerate()
    fake_requests = types.SimpleNamespace(post=None, exceptions=types.SimpleNamespace(Timeout=Exception, ConnectionError=Exception))
    gid.requests = fake_requests

    # ---- 1. generation JSON path --------------------------------------------
    print("[1] generation payload")
    CAPTURED.clear()
    usage = {"total_tokens": 1780, "input_tokens": 24, "output_tokens": 1756,
             "input_tokens_details": {"text_tokens": 24, "image_tokens": 0}}
    fake_requests.post = make_fake_post(usage=usage, w=96, h=64)
    imgs, masks, cost_info, cache_key = node.generate(
        prompt="a red chair", model="gpt-image-2.5-flare", quality="high", size="1536x1024",
        custom_width=1024, custom_height=1024, background="auto", output_format="png",
        output_compression=90, moderation="low", n=1, seed=7)
    req = CAPTURED[-1]
    check(req["url"].endswith("/images/generations"), "generations endpoint")
    check(req["headers"]["Authorization"] == "Bearer sk-test-not-real", "bearer header from env")
    check(req["json"]["model"] == "gpt-image-2.5-flare" and req["json"]["quality"] == "high", "model/quality sent")
    check(req["json"]["size"] == "1536x1024", "size sent")
    check("background" not in req["json"], "background=auto omitted")
    check("output_compression" not in req["json"], "no compression for png")
    check(req["files"] is None and req["data"] is None, "json body, not multipart")
    check(imgs.shape == (1, 64, 96, 3) and imgs.dtype == torch.float32, f"IMAGE shape {tuple(imgs.shape)}")
    check(masks.shape == (1, 64, 96) and float(masks.max()) == 0.0, "opaque -> zero mask")
    expected = (24 * 5.0 + 1756 * 30.0) / 1e6
    check(cost_info.startswith(f"${expected:.4f} USD (exact)"), f"exact cost {expected:.4f} in cost_info")
    check("gpt-image-2.5-flare|high|1536x1024|auto|png|1|7||a red chair" == cache_key, "cache_key deterministic")

    # ---- 2. cost math + negative control -------------------------------------
    print("[2] cost math")
    usd, tok = gid.compute_cost("gpt-image-2.5-sunburst", {"input_tokens": 1100, "output_tokens": 196,
                                "input_tokens_details": {"text_tokens": 100, "image_tokens": 1000}})
    check(abs(usd - (100 * 5 + 1000 * 8 + 196 * 30) / 1e6) < 1e-9, "sunburst edit cost with image input")
    usd_mini, _ = gid.compute_cost("gpt-image-1-mini", {"input_tokens": 100, "output_tokens": 196,
                                   "input_tokens_details": {"text_tokens": 100, "image_tokens": 0}})
    check(usd_mini < usd, "mini cheaper than sunburst (negative control on rate table)")
    usd_cached, _ = gid.compute_cost("gpt-image-2", {"input_tokens": 1100, "output_tokens": 0,
                                     "input_tokens_details": {"text_tokens": 100, "image_tokens": 1000, "cached_tokens": 1000}})
    check(abs(usd_cached - (100 * 5 + 1000 * 1.25) / 1e6) < 1e-9, "cached image tokens billed at cached rate")
    wrong, _ = gid.compute_cost("gpt-image-2.5-flare", {"output_tokens": 196})
    check(abs(wrong - 0.00588) < 1e-9 and wrong != expected, "negative control: different usage -> different cost")

    # ---- 3. edits multipart path + mask ---------------------------------------
    print("[3] edits + mask")
    CAPTURED.clear()
    fake_requests.post = make_fake_post(usage=usage)
    ref = torch.rand(1, 300, 200, 3)
    m = torch.zeros(1, 150, 100)
    m[0, :75, :] = 1.0  # top half repaint
    imgs, masks, cost_info, cache_key = node.generate(
        prompt="make it blue", model="gpt-image-2.5-sunburst", quality="medium", size="auto",
        custom_width=1024, custom_height=1024, background="auto", output_format="jpeg",
        output_compression=80, moderation="auto", n=1, seed=1, images=ref, mask=m,
        reference_max_size="1024", input_fidelity="high")
    req = CAPTURED[-1]
    check(req["url"].endswith("/images/edits"), "edits endpoint when refs attached")
    check(req["json"] is None and req["data"]["model"] == "gpt-image-2.5-sunburst", "multipart data fields")
    check(req["data"]["input_fidelity"] == "high", "input_fidelity forwarded")
    check(req["data"]["output_compression"] == 80, "compression sent for jpeg")
    check("size" not in req["data"], "size=auto omitted")
    names = [f[0] for f in req["files"]]
    check(names == ["image", "mask"], f"single ref uses 'image' field + mask: {names}")
    mask_png = Image.open(io.BytesIO(req["files"][1][1][1]))
    check(mask_png.mode == "RGBA" and mask_png.size == (200, 300), f"mask RGBA resized to ref {mask_png.size}")
    a = np.array(mask_png.getchannel("A"))
    check(a[10, 10] == 0 and a[290, 10] == 255, "alpha 0 in repaint region, 255 elsewhere")
    check(cache_key.split("|")[7] != "", "cache_key carries ref hash")

    # ---- 4. multi-ref field name + batch + downscale ---------------------------
    print("[4] multi-ref")
    CAPTURED.clear()
    fake_requests.post = make_fake_post(usage=usage, n=2)
    ref3 = torch.rand(3, 2600, 1300, 3)
    imgs, masks, _, _ = node.generate(
        prompt="combine", model="gpt-image-2", quality="low", size="1024x1024",
        custom_width=1024, custom_height=1024, background="auto", output_format="png",
        output_compression=90, moderation="low", n=2, seed=1, images=ref3, reference_max_size="1024")
    req = CAPTURED[-1]
    names = [f[0] for f in req["files"]]
    check(names == ["image[]"] * 3, f"multi ref uses 'image[]': {names}")
    up = Image.open(io.BytesIO(req["files"][0][1][1]))
    check(max(up.size) == 1024, f"ref downscaled to max edge 1024: {up.size}")
    check(imgs.shape[0] == 2 and masks.shape[0] == 2, "n=2 -> batch of 2")

    # ---- 4b. per-socket refs mixed with batch, order + sizes preserved ----------
    print("[4b] sockets")
    CAPTURED.clear()
    fake_requests.post = make_fake_post(usage=usage)
    node.generate(
        prompt="mix", model="gpt-image-2.5-flare", quality="low", size="auto",
        custom_width=1024, custom_height=1024, background="auto", output_format="png",
        output_compression=90, moderation="low", n=1, seed=1,
        image_1=torch.rand(1, 100, 50, 3), image_3=torch.rand(1, 60, 80, 3),
        images=torch.rand(2, 40, 40, 3), reference_max_size="original")
    req = CAPTURED[-1]
    sizes = [Image.open(io.BytesIO(f[1][1])).size for f in req["files"]]
    check(sizes == [(50, 100), (80, 60), (40, 40), (40, 40)], f"socket order + mixed sizes kept: {sizes}")
    try:
        node.generate(prompt="x", model="gpt-image-2.5-flare", quality="low", size="auto",
                      custom_width=1024, custom_height=1024, background="auto", output_format="png",
                      output_compression=90, moderation="low", n=1, seed=1,
                      image_1=torch.rand(1, 8, 8, 3), images=torch.rand(16, 8, 8, 3))
        check(False, "17 refs should be rejected")
    except ValueError:
        check(True, "17 refs rejected")

    # ---- 5. fallbacks + validation --------------------------------------------
    print("[5] fallbacks")
    check(gid._resolve_quality("gpt-image-1.5", "max") == "high", "legacy model max -> high")
    check(gid._resolve_quality("gpt-image-2.5-flare", "max") == "max", "2.5 keeps max")
    check(gid._resolve_size("gpt-image-1-mini", "2048x2048", 0, 0) == "1024x1024", "legacy 2048 -> 1024")
    check(gid._resolve_size("gpt-image-2.5-flare", "custom", 1920, 1088) == "1920x1088", "custom ok")
    for bad in [(1000, 1000), (3840, 1024), (480, 480), (4096, 2048)]:
        try:
            gid._resolve_size("gpt-image-2.5-flare", "custom", *bad)
            check(False, f"custom {bad} should be rejected")
        except ValueError:
            check(True, f"custom {bad} rejected")
    try:
        node.generate(prompt="x", model="gpt-image-2.5-flare", quality="low", size="auto",
                      custom_width=1024, custom_height=1024, background="auto", output_format="png",
                      output_compression=90, moderation="low", n=1, seed=1, mask=torch.zeros(1, 8, 8))
        check(False, "mask without image should raise")
    except ValueError:
        check(True, "mask without image rejected")

    # ---- 6. transparent alpha -> mask, jpeg forced to png ---------------------
    print("[6] transparency")
    CAPTURED.clear()
    fake_requests.post = make_fake_post(usage=usage, rgba=True)
    imgs, masks, _, _ = node.generate(
        prompt="logo", model="gpt-image-2.5-flare", quality="low", size="1024x1024",
        custom_width=1024, custom_height=1024, background="transparent", output_format="jpeg",
        output_compression=90, moderation="low", n=1, seed=1)
    check(CAPTURED[-1]["json"]["output_format"] == "png", "jpeg+transparent forced to png")
    check(CAPTURED[-1]["json"]["background"] == "transparent", "background sent")
    check(abs(float(masks.mean()) - (1 - 128 / 255)) < 1e-3, "alpha 128 -> mask 0.498")

    # ---- 7. error surfacing ----------------------------------------------------
    print("[7] errors")
    fake_requests.post = make_fake_post(status=401)
    try:
        node.generate(prompt="x", model="gpt-image-2.5-flare", quality="low", size="auto",
                      custom_width=1024, custom_height=1024, background="auto", output_format="png",
                      output_compression=90, moderation="low", n=1, seed=1)
        check(False, "401 should raise")
    except RuntimeError as e:
        msg = str(e)
        check("401" in msg and "org_verification_required" in msg and "verified" in msg, "401 surfaces code + verification hint")

    # ---- 8. no usage -> estimated ----------------------------------------------
    print("[8] missing usage")
    fake_requests.post = make_fake_post(usage=None)
    _, _, cost_info, _ = node.generate(
        prompt="x", model="gpt-image-2.5-flare", quality="low", size="1024x1024",
        custom_width=1024, custom_height=1024, background="auto", output_format="png",
        output_compression=90, moderation="low", n=1, seed=1)
    check("(estimated)" in cost_info, "falls back to estimate label without usage")


def run_live():
    import requests as real_requests
    gid.requests = real_requests
    try:
        import folder_paths
        gid.folder_paths = folder_paths
    except ImportError:
        pass
    os.environ.pop("OPENAI_API_KEY", None) if os.environ.get("OPENAI_API_KEY") == "sk-test-not-real" else None
    node = gid.GPTImageGenerate()
    print("[live] one gpt-image-2.5-flare low 1024x1024 call")
    imgs, masks, cost_info, _ = node.generate(
        prompt="a single matte black ceramic cup on raw concrete, soft north light",
        model="gpt-image-2.5-flare", quality="low", size="1024x1024",
        custom_width=1024, custom_height=1024, background="auto", output_format="png",
        output_compression=90, moderation="low", n=1, seed=1)
    check(imgs.shape[1:] == (1024, 1024, 3), f"live image shape {tuple(imgs.shape)}")
    check("(exact)" in cost_info, "live usage present")
    print("  ", cost_info)


if __name__ == "__main__":
    run_offline()
    if "--live" in sys.argv:
        run_live()
    print(f"\n{PASSES} passed, {len(FAILS)} failed")
    if FAILS:
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
