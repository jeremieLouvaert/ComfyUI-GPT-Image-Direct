# ComfyUI-GPT-Image-Direct

**OpenAI GPT Image 2.5 (Flare and Sunburst), GPT Image 2, 1.5 and 1-mini in ComfyUI, on your own API key.**

No Comfy credits, no middleman, no markup. One node, text-to-image and reference editing, exact USD cost printed after every run.

## Why

ComfyUI's built-in OpenAI partner node bills through Comfy.org credits. Its own price badge quotes about 1.4x OpenAI's list price, and the credit table on docs.comfy.org implies more than that once you buy credits in packs. This node sends the same request to the same OpenAI endpoint with your key, and reads the real token usage back so you see what each image actually cost.

## Node

| Node | What it does |
|------|--------------|
| **GPT Image Generate (Direct API)** | Text-to-image, or edit up to 16 reference images with an optional inpaint mask. Outputs the image batch, an alpha mask (for transparent backgrounds), a `cost_info` string and a `cache_key`. |

Category: `AKURATE/GPT Image Direct`

### Inputs

- `model`: `gpt-image-2.5-flare` (fast, everyday), `gpt-image-2.5-sunburst` (slower, more precise text and layouts), `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1-mini`. Flare and Sunburst cost the same per token.
- `quality`: `low`, `medium`, `high`, `xhigh`, `max`, `auto`. This is the cost lever. Rough 1024x1024 prices at list rate: low ~$0.006, medium ~$0.02, high ~$0.05, max ~$0.21. `xhigh` and `max` exist on 2.5 only; other models fall back to `high`.
- `size`: presets up to 3840x2160 or `custom` (multiples of 16, aspect 1:3 to 3:1, 480 to 3840 px per edge). Legacy models fall back to 1024x1024 for sizes they lack.
- `background`: `auto`, `opaque`, `transparent` (forces png or webp; the alpha lands on the `mask` output).
- `output_format` and `output_compression`: png, jpeg or webp. File format only, no effect on cost.
- `moderation`: `auto` or `low`.
- `n`: images per run, 1 to 8.
- `seed`: not sent to OpenAI (the API has no seed). Changing it forces a re-run and a fresh `cache_key`.
- `image_1` to `image_4` (optional): individual reference sockets, any sizes. `images` (optional): a batch appended after them. 16 references total. Any reference switches to the edits endpoint.
- `mask` (optional): white = repaint. Needs exactly one reference image. Auto-resized to the reference.
- `reference_max_size`: downscale references before upload (2048 default). Smaller references mean fewer input tokens.
- `input_fidelity`: `auto` (not sent), `low`, `high`. Edits only. High keeps more of the reference but costs more input tokens.
- `api_key`, `api_base`, `timeout_sec`: override the key, point at an OpenAI-compatible base URL (Azure, fal, a proxy), and give slow max-quality 4K runs more time.

### Outputs

- `images`: IMAGE batch.
- `mask`: MASK, 1 where the image is transparent (same convention as Load Image).
- `cost_info`: `"$0.0528 USD (exact) | gpt-image-2.5-flare high 1536x1024 x1 | 14.2s | tokens: 24 text-in, 0 img-in, 0 cached, 1756 out | gpt_image_20260919_120000_00.png"`. Exact when the API returns `usage`, otherwise labelled `(estimated)`.
- `cache_key`: deterministic string of every input, for Hash Vault in ComfyUI-API-Optimizer.

Images are also saved to `output/gpt_image_direct/`.

## Installation

1. Copy this folder into `ComfyUI/custom_nodes/` (or clone it there).
2. No extra pip installs. It uses `requests` and Pillow, which ComfyUI already ships.
3. Restart ComfyUI.

## API key

Resolution order:

1. The `api_key` input on the node
2. `OPENAI_API_KEY` environment variable
3. `openai_api_key.txt` in your ComfyUI root (next to `main.py`)

Get a key at https://platform.openai.com/api-keys. GPT Image models need a one-time Organization Verification at https://platform.openai.com/settings/organization/general. Without it the node reports the 401/403 with a pointer to that page.

## Keeping it cheap

- `quality=low` for iteration, `medium` for most finals. `high` is 9x low, `max` is 35x low.
- Keep references at 1024 unless the edit needs detail. Every reference image costs input tokens on every call.
- Leave `input_fidelity` on `auto` unless the reference must survive pixel-exact.
- Wide and tall presets (1536x1024, 1024x1536) often cost slightly less than square at the same quality.
- For big offline batches, OpenAI's Batch API is half price. That is a separate async workflow, not this node.

## Pricing reference

USD per 1M tokens, OpenAI list price as of 2026-09-19:

| Model | Text in | Cached in | Image in | Image out |
|-------|---------|-----------|----------|-----------|
| gpt-image-2.5-flare / sunburst | 5.00 | 1.25 | 8.00 | 30.00 |
| gpt-image-2 | 5.00 | 1.25 | 8.00 | 30.00 |
| gpt-image-1.5 | 5.00 | 1.25 | 8.00 | 32.00 |
| gpt-image-1-mini | 2.00 | 0.20 | 2.50 | 8.00 |

The node multiplies the `usage` counts the API returns by this table. Update `PRICE_PER_1M` in `gpt_image_direct_nodes.py` if OpenAI changes prices.

## Testing

```
python tools/test_gpt_image_direct.py          # offline, mocked HTTP, no spend
python tools/test_gpt_image_direct.py --live   # plus one real low-quality Flare call (~$0.006)
```

## License

MIT
