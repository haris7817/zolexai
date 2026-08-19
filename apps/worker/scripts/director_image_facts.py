"""Describes what an uploaded photograph visibly shows, for I2V Director mode.

Executed by the worker in the LTX environment (`uv run python`, cwd = the LTX
repo), exactly like `director_plan.py`. The contract is the same small pipe:

    stdin   one JSON object:
              gemma_root      HF directory of the multimodal instruct checkpoint
              image_path      the staged source image on local disk
              system_prompt   what to look for (owned by the WORKER, so it
                              versions and is tested with worker code)
              user_prompt     the request text accompanying the image
              max_new_tokens  decode budget
              begin_marker /  the model's raw completion is echoed between
              end_marker      these, so the worker can find it in any noise
                              the model libraries print around it

    stdout  logs, then the completion between the markers
    exit    0 when a completion was produced; non-zero when this checkpoint
            cannot do the job at all (text-only weights, missing processor)

The failure posture matters more than the success path: whether the on-box
checkpoint accepts image input is UNMEASURED, and the worker treats a non-zero
exit as "no facts today" rather than as a failed job. `AutoProcessor` +
`AutoModelForImageTextToText` raise quickly on a text-only checkpoint, which is
precisely the honest answer this script exists to deliver cheaply.
"""

from __future__ import annotations

import json
import sys
import time


def main() -> int:
    request = json.loads(sys.stdin.read())
    root = request["gemma_root"]
    image_path = request["image_path"]
    max_new_tokens = int(request.get("max_new_tokens", 400))

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    started = time.time()
    processor = AutoProcessor.from_pretrained(root)
    model = AutoModelForImageTextToText.from_pretrained(
        root, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"vision model loaded in {time.time() - started:.1f}s", flush=True)

    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": request["system_prompt"]}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": request["user_prompt"]},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    started = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    completion = processor.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"described in {time.time() - started:.1f}s", flush=True)

    print(request.get("begin_marker", "===IMAGE_FACTS_BEGIN==="), flush=True)
    print(completion, flush=True)
    print(request.get("end_marker", "===IMAGE_FACTS_END==="), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
