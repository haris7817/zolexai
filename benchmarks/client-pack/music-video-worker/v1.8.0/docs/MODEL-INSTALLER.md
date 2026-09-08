# Automatic model installer

The model weights are intentionally installed on the backend instead of embedded in the developer ZIP. The current required set is approximately 91 GB before filesystem and download-cache overhead: the six LTX-2.5 files used by the guided two-stage renderer are about 80 GB, Qwen2.5-VL-3B is 7.52 GB, and Faster-Whisper large-v3 is 3.09 GB.

LTX-2.5 is gated under the LTX community license. A human must review and accept the terms on the official model page before an account token can download the files:

<https://huggingface.co/Lightricks/LTX-2.5>

## Install all required models

Install the package and downloader dependency, authenticate without placing the token in the command history, then run the installer:

```bash
python -m pip install '.[api,lyrics,reference-links,reference-ai,models]'
hf auth login
zolex-music-video install-models --models-root /models
```

The final command performs these operations automatically:

1. Confirms that the authenticated account can access the gated LTX files before downloading anything.
2. Resolves and records the exact repository revision used for every model.
3. Calculates the required download size and checks free disk space with a safety margin.
4. Downloads only the six LTX files used by this worker, the complete Faster-Whisper runtime model, and the complete Qwen vision model.
5. Reuses complete local files and resumes downloads through the Hugging Face cache.
6. Compares every local file with its official remote size and verifies every LFS file against its SHA-256 digest.
7. Writes `/models/zolex-model-install-manifest.json` with revisions, file sizes and verification results.
8. Writes `/models/zolex-models.env` with all LTX, Whisper and Qwen paths. It never writes the Hugging Face token into either file.

Load the generated values through the backend process manager. For a shell smoke test:

```bash
set -a
. /models/zolex-models.env
set +a
zolex-music-video doctor
```

`LTX_PYTHON` still points to the separate official LTX environment, for example `/srv/ltx/.venv/bin/python`.

## Preflight and selective installation

Check access, exact revisions and required disk space without downloading:

```bash
zolex-music-video install-models --models-root /models --dry-run
```

If one model is already managed elsewhere, skip it:

```bash
zolex-music-video install-models --models-root /models --skip-ltx
zolex-music-video install-models --models-root /models --skip-whisper
zolex-music-video install-models --models-root /models --skip-qwen
```

Use `--size-only` only when installation speed matters more than the additional local SHA-256 pass. The Hugging Face downloader still performs its normal content-addressed transfer checks, and the installer still compares remote and local sizes.

## Official repositories

- LTX-2.5: <https://huggingface.co/Lightricks/LTX-2.5>
- Faster-Whisper large-v3: <https://huggingface.co/Systran/faster-whisper-large-v3>
- Qwen2.5-VL-3B-Instruct: <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>

Do not redistribute a prebuilt weight archive without independently confirming every model's current license and the service's eligibility. Keeping weights on backend model storage also allows workflow-code updates without repeatedly transferring approximately 91 GB.
