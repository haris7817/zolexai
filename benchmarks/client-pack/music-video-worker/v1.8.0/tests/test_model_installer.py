from __future__ import annotations

import unittest
import fnmatch
import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from zolex_music_worker.model_installer import (
    LTX_FILES,
    MODEL_BUNDLES,
    _safe_models_root,
    _selected_remote_files,
    build_model_env,
    install_required_models,
)


class ModelInstallerTests(unittest.TestCase):
    def test_ltx_bundle_contains_every_path_used_by_the_worker(self) -> None:
        bundle = next(item for item in MODEL_BUNDLES if item.key == "ltx")
        self.assertEqual(bundle.allow_patterns, LTX_FILES)
        info = SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=name) for name in LTX_FILES]
        )
        selected = _selected_remote_files(info, bundle)
        self.assertEqual([item.rfilename for item in selected], list(LTX_FILES))

    def test_generated_environment_points_every_backend_to_local_models(self) -> None:
        text = build_model_env(Path("/models"), ("ltx", "whisper", "qwen"))
        self.assertIn("LTX_TRANSFORMER_PATH=/models/ltx-2.5/diffusion_models/", text)
        self.assertIn("ZOLEX_TRANSCRIPTION_MODEL=/models/faster-whisper-large-v3", text)
        self.assertIn("ZOLEX_REFERENCE_VISION_MODEL=/models/qwen2.5-vl-3b-instruct", text)
        self.assertIn("ZOLEX_REFERENCE_VISION_LOCAL_FILES_ONLY=true", text)
        self.assertNotIn("HF_TOKEN", text)

    def test_filesystem_root_is_rejected_as_a_model_target(self) -> None:
        with self.assertRaisesRegex(Exception, "filesystem root"):
            _safe_models_root("/")

    def test_complete_installer_contract_without_network(self) -> None:
        contents: dict[tuple[str, str], bytes] = {}
        infos: dict[str, SimpleNamespace] = {}
        for bundle in MODEL_BUNDLES:
            filenames = list(bundle.allow_patterns) if bundle.key == "ltx" else ["config.json", "weights.bin"]
            siblings = []
            for index, filename in enumerate(filenames):
                content = f"{bundle.key}-{index}".encode()
                contents[(bundle.repo_id, filename)] = content
                siblings.append(
                    SimpleNamespace(
                        rfilename=filename,
                        size=len(content),
                        lfs={"size": len(content), "sha256": hashlib.sha256(content).hexdigest()},
                    )
                )
            infos[bundle.repo_id] = SimpleNamespace(sha=bundle.key * 8, siblings=siblings)

        hub = types.ModuleType("huggingface_hub")
        utils = types.ModuleType("huggingface_hub.utils")

        class FakeHubError(OSError):
            pass

        class FakeApi:
            def __init__(self, token=None):
                self.token = token

            def model_info(self, repo_id, revision, files_metadata):
                self.assertions = (revision, files_metadata)
                return infos[repo_id]

        def fake_snapshot_download(*, repo_id, revision, local_dir, allow_patterns, token, max_workers):
            del revision, token, max_workers
            for (source_repo, filename), content in contents.items():
                if source_repo != repo_id:
                    continue
                if not any(fnmatch.fnmatch(filename, pattern) for pattern in allow_patterns):
                    continue
                path = Path(local_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            return str(local_dir)

        hub.HfApi = FakeApi
        hub.get_token = lambda: "test-token-never-written"
        hub.get_hf_file_metadata = lambda *args, **kwargs: SimpleNamespace()
        hub.hf_hub_url = lambda repo_id, filename, revision: f"https://example/{repo_id}/{revision}/{filename}"
        hub.snapshot_download = fake_snapshot_download
        utils.GatedRepoError = FakeHubError
        utils.HfHubHTTPError = FakeHubError

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            with patch.dict(
                sys.modules,
                {"huggingface_hub": hub, "huggingface_hub.utils": utils},
            ):
                result = install_required_models(models_root=root, max_workers=2)
            self.assertEqual(result["status"], "complete")
            manifest_text = (root / "zolex-model-install-manifest.json").read_text()
            manifest = json.loads(manifest_text)
            self.assertEqual(len(manifest["models"]), 3)
            self.assertNotIn("test-token-never-written", manifest_text)
            self.assertTrue((root / "zolex-models.env").is_file())


if __name__ == "__main__":
    unittest.main()
