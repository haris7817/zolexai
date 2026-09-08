from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .adapters import AnchorAdapter
from .models import MusicVideoRequest
from .utils import atomic_write_json, ensure_existing_file


def _generated_identity_prompt(performer_id: str, treatment: dict[str, Any]) -> str:
    relationship = "romantic lead" if treatment.get("genre", "").startswith("romantic") else "music-video lead"
    return (
        f"Create the canonical identity portrait for fictional adult {performer_id}, a {relationship}. "
        "Show exactly one person, age 25 to 35, in a photorealistic neutral waist-up portrait facing camera at eye level. "
        "Use a distinctive natural face, visible skin texture, stable hairstyle, relaxed expression, simple unbranded wardrobe, "
        "soft daylight, a plain gray background and clear separation around the head and shoulders. "
        "This image will be the locked identity reference for all later scenes. No celebrity likeness, extra people, split panels, "
        "text, logos, watermark, dramatic pose, cropped head, malformed hands or accessories that obscure the face."
    )


def build_identity_package(
    *,
    request: MusicVideoRequest,
    treatment: dict[str, Any],
    output_dir: Path,
    adapter: AnchorAdapter,
    timeout: int,
) -> dict[str, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    supplied = {performer.id: performer for performer in request.performers}
    package: dict[str, list[Path]] = {}
    manifest: dict[str, Any] = {"schema_version": 1, "performers": []}

    for performer_id in treatment.get("performer_ids", []):
        performer = supplied.get(performer_id)
        references = (
            [ensure_existing_file(path, f"Reference for {performer_id}") for path in performer.reference_images]
            if performer
            else []
        )
        source = "supplied"
        if not references:
            source = "generated" if performer is None else "generated_from_description"
            output = output_dir / f"{performer_id}-canonical.png"
            payload = {
                "schema_version": 1,
                "anchor_role": "identity_reference",
                "format": "1:1",
                "width": 768,
                "height": 768,
                "performer_ids": [performer_id],
                "references": [],
                "prompt": _generated_identity_prompt(performer_id, treatment),
                "output": str(output),
            }
            if not output.exists():
                adapter.generate(
                    request_payload=payload,
                    output=output,
                    width=768,
                    height=768,
                    references=[],
                    timeout=timeout,
                    log_path=output.with_suffix(".command.json"),
                )
            with Image.open(output) as image:
                if image.size != (768, 768):
                    raise ValueError(
                        f"Generated identity reference {output} is {image.size[0]}x{image.size[1]}, expected 768x768"
                    )
                image.verify()
            references = [output.resolve()]
        package[performer_id] = references
        manifest["performers"].append(
            {
                "performer_id": performer_id,
                "source": source,
                "references": [str(path) for path in references],
                "role": performer.role if performer else "generated_lead",
                "description": performer.description if performer else "fictional adult identity",
            }
        )

    atomic_write_json(output_dir / "identity-package.json", manifest)
    return package
