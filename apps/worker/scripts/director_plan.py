"""Runs one Director-mode planning request on a local instruct model.

Executed by the worker in the LTX environment (`uv run python`, cwd = the LTX
repo), the way `person_matte.py` is: torch and transformers live there, the
worker deliberately has neither. The contract is a small, stable pipe:

    stdin   one JSON object:
              gemma_root      HF directory of the instruct checkpoint
              system_prompt   the planning brief (owned by the WORKER, so it
                              versions and is tested with worker code — this
                              script knows nothing about plan shapes)
              user_prompt     the rendered request
              sample          false = greedy (deterministic first attempt),
                              true = sampled retry
              seed            sampling seed
              max_new_tokens  decode budget
              begin_marker /  the model's raw completion is echoed between
              end_marker      these, so the worker can find it in any noise
                              the model libraries print around it

    stdout  logs, then the completion between the markers
    exit    0 when a completion was produced (validity is the worker's call)

Greedy decoding for the first attempt mirrors the LTX runtime's own choice for
its Gemma-4 enhancer; the retry samples because regenerating a refused plan
greedily would reproduce it token for token.
"""

from __future__ import annotations

import json
import sys
import time


def main() -> int:
    request = json.loads(sys.stdin.read())
    root = request["gemma_root"]
    sample = bool(request.get("sample"))
    seed = int(request.get("seed", 0))
    max_new_tokens = int(request.get("max_new_tokens", 1600))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(root)
    model = AutoModelForCausalLM.from_pretrained(
        root, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"planner model loaded in {time.time() - started:.1f}s", flush=True)

    messages = [
        {"role": "system", "content": request["system_prompt"]},
        {"role": "user", "content": request["user_prompt"]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")

    decode: dict[str, object] = {"max_new_tokens": max_new_tokens}
    if sample:
        torch.manual_seed(seed)
        decode.update({"do_sample": True, "temperature": 0.7, "top_p": 0.9})
    else:
        decode["do_sample"] = False

    started = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, **decode)
    completion = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(
        f"planned in {time.time() - started:.1f}s "
        f"({output.shape[1] - inputs['input_ids'].shape[1]} tokens, sample={sample})",
        flush=True,
    )

    print(request.get("begin_marker", "===DIRECTOR_PLAN_BEGIN==="), flush=True)
    print(completion, flush=True)
    print(request.get("end_marker", "===DIRECTOR_PLAN_END==="), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
