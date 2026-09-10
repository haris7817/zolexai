"""Shared BiRefNet helpers for the V2V subprocess scripts."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonComponent:
    mask: object
    x: int
    y: int
    width: int
    height: int
    area: int

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2


class PersonSegmenter:
    """Loads a local BiRefNet checkpoint and returns person foreground masks."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
        except ImportError as error:
            raise RuntimeError(
                "V2V segmentation dependencies are missing; run install_v2v_models.sh"
            ) from error

        root = Path(
            model_dir
            or os.environ.get("ZOLEXAI_BIREFNET_MODEL", "")
            or "/workspace/ltx2-benchmark/models/BiRefNet"
        )
        if not root.exists():
            raise RuntimeError(
                f"BiRefNet checkpoint is missing at {root}; run install_v2v_models.sh"
            )
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = (
            AutoModelForImageSegmentation.from_pretrained(
                str(root), trust_remote_code=True, local_files_only=True
            )
            .eval()
            .to(self.device)
        )
        #: The checkpoint's own precision, read off the loaded weights rather
        #: than assumed. The published BiRefNet weights are fp16, and feeding
        #: them an fp32 tensor fails on the first convolution with "Input type
        #: (float) and bias type (c10::Half) should be the same" — measured on
        #: the RTX PRO 6000, 10 Sep 2026, the first time this ran on real
        #: hardware. Reading the dtype works for either precision, where
        #: hard-coding `.half()` would break the day the weights change.
        self.dtype = next(self.model.parameters()).dtype

    def mask(self, image):
        import numpy as np
        from PIL import Image
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        from torchvision.transforms.functional import InterpolationMode

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        original_size = image.size
        transform = Compose(
            [
                Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(self.device, self.dtype)
        with self.torch.inference_mode():
            output = self.model(tensor)
        prediction = self._prediction_tensor(output).sigmoid().float()
        while prediction.ndim > 2:
            prediction = prediction[0]
        prediction = prediction.detach().cpu().numpy()
        lo, hi = float(prediction.min()), float(prediction.max())
        if hi - lo > 1e-6:
            prediction = (prediction - lo) / (hi - lo)
        mask = Image.fromarray((prediction * 255).clip(0, 255).astype(np.uint8))
        return np.asarray(mask.resize(original_size, Image.Resampling.BILINEAR))

    def _prediction_tensor(self, output):
        tensors = list(self._tensors(output))
        if not tensors:
            raise RuntimeError("BiRefNet returned no segmentation tensor")
        # BiRefNet exposes its final high-resolution prediction last.
        return tensors[-1]

    def _tensors(self, value):
        if self.torch.is_tensor(value):
            yield value
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from self._tensors(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from self._tensors(item)
            return
        for attribute in ("logits", "predictions"):
            item = getattr(value, attribute, None)
            if item is not None:
                yield from self._tensors(item)


def components(mask, *, minimum_area_ratio: float = 0.008) -> list[PersonComponent]:
    """Connected foreground regions, largest-first then addressable left-right."""
    import cv2
    import numpy as np

    binary = np.where(mask >= 128, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum = max(64, round(binary.shape[0] * binary.shape[1] * minimum_area_ratio))
    found: list[PersonComponent] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < minimum:
            continue
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        found.append(PersonComponent(component_mask, x, y, width, height, area))
    return sorted(found, key=lambda component: component.area, reverse=True)


def run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(" | ".join(detail) or f"command exited {completed.returncode}")


def cover(image, width: int, height: int):
    """Resize and centre-crop a PIL image to exactly width × height."""
    from PIL import Image

    image = image.convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


#: How much of the source person's box the cutout may fill. Slightly inside it
#: so the new person never overhangs the silhouette the control signal is
#: about to enforce.
FIT_WIDTH, FIT_HEIGHT = 0.90, 0.96

#: How close to the reference photo's bottom edge the matte must reach before
#: the subject counts as TRUNCATED — a crop of a person rather than a whole
#: one. As a fraction of the reference's height.
REFERENCE_TRUNCATION_TOLERANCE = 0.02

#: The fraction of a matte's height treated as "the head", measured down from
#: its topmost pixel.
HEAD_BAND = 0.12


def head_band(mask, box: tuple[int, int, int, int]) -> tuple[int, int] | None:
    """(x0, x1) of the matte across the top `HEAD_BAND` of its height.

    None when the band holds nothing, which means the matte is not shaped like
    a person and the caller should not trust it.
    """
    import numpy as np

    x0, y0, x1, y1 = box
    band = max(1, round((y1 - y0) * HEAD_BAND))
    sub = mask[y0 : y0 + band, x0:x1] >= 128
    columns = np.any(sub, axis=0)
    if not columns.any():
        return None
    left = x0 + int(np.argmax(columns))
    right = x0 + len(columns) - int(np.argmax(columns[::-1]))
    if right - left < 2:
        return None
    return left, right


def is_truncated(box: tuple[int, int, int, int], height: int) -> bool:
    """Does the reference's subject run off the bottom of their own photo?

    If it does, the cutout's lowest row is a crop edge and not a pair of feet,
    which is what decides how the cutout may be placed.
    """
    return box[3] >= height - max(1, round(REFERENCE_TRUNCATION_TOLERANCE * height))


def place_cutout(
    source_box: tuple[int, int, int, int],
    cutout_size: tuple[int, int],
    *,
    truncated: bool,
    source_head: tuple[int, int] | None = None,
    reference_head: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """How big the reference cutout should be, and where it goes.

    Carried over from `scripts/person_anchor.py`, where it was measured on 20
    Aug 2026, because the client's multi-person anchor arrived bottom-aligning
    every cutout unconditionally — the exact defect that fix exists for.

    Two cases, and they differ in which part of the person is trustworthy:

    * A WHOLE reference figure shares its ground with the source person, so it
      is fitted inside their box and bottom-aligned — their feet meet the same
      floor.
    * A TRUNCATED one (a headshot, a bust) has no feet to align. Bottom-
      aligning it plants the bust at the source person's feet and scales it to
      their width, which anchored a full-body dance source on a disembodied
      bust sitting on the road and put a woman standing in the street through
      the whole video. So it is scaled to match the source person's HEAD and
      hung from the top of their box; its body continues past the bottom, and
      the control signal states the body's pose for every frame regardless.

    Head matching rather than box-width matching, because the box is not
    reliably a person: BiRefNet mattes the salient OBJECT, and on someone
    standing beside a car it returns the person and the car as one region.
    Both head bands are needed for it; if either is missing, the box width is
    the fallback.
    """
    sx0, sy0, sx1, sy1 = source_box
    box_w, box_h = sx1 - sx0, sy1 - sy0
    cut_w, cut_h = cutout_size

    if not truncated:
        scale = min(FIT_WIDTH * box_w / cut_w, FIT_HEIGHT * box_h / cut_h)
        size = (max(1, round(cut_w * scale)), max(1, round(cut_h * scale)))
        return size, (sx0 + (box_w - size[0]) // 2, sy1 - size[1])

    if source_head and reference_head:
        source_width = source_head[1] - source_head[0]
        reference_width = reference_head[1] - reference_head[0]
        scale = source_width / max(1, reference_width)
        size = (max(1, round(cut_w * scale)), max(1, round(cut_h * scale)))
        # Line the two heads up horizontally, rather than the two boxes.
        reference_centre = (reference_head[0] + reference_head[1]) / 2 * scale
        paste_x = round((source_head[0] + source_head[1]) / 2 - reference_centre)
        return size, (paste_x, sy0)

    scale = FIT_WIDTH * box_w / cut_w
    size = (max(1, round(cut_w * scale)), max(1, round(cut_h * scale)))
    return size, (sx0 + (box_w - size[0]) // 2, sy0)


def composite_person(base, reference, reference_mask, target: PersonComponent):
    """Place one reference person into one source-person bounding box.

    WHICH EDGE the cutout is aligned by depends on whether the photo shows a
    whole person or a crop of one — see `place_cutout`. A headshot is the
    commonest thing a customer uploads, so that is the common case here, not
    an edge case.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageFilter

    ys, xs = np.nonzero(reference_mask >= 128)
    if len(xs) == 0:
        raise RuntimeError("reference contains no segmented person")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    person = reference.crop((x0, y0, x1, y1)).convert("RGB")
    alpha = Image.fromarray(reference_mask[y0:y1, x0:x1]).convert("L")

    source_box = (target.x, target.y, target.x + target.width, target.y + target.height)
    reference_box = (x0, y0, x1, y1)
    # Head bands are measured in each image's own coordinates; the reference's
    # is rebased onto the cutout, which is what gets scaled.
    reference_head = head_band(reference_mask, reference_box)
    if reference_head is not None:
        reference_head = (reference_head[0] - x0, reference_head[1] - x0)
    size, (left, top) = place_cutout(
        source_box,
        (person.width, person.height),
        truncated=is_truncated(reference_box, reference.height),
        source_head=head_band(target.mask, source_box),
        reference_head=reference_head,
    )

    person = person.resize(size, Image.Resampling.LANCZOS)
    alpha = alpha.resize(size, Image.Resampling.LANCZOS)
    alpha_np = cv2.dilate(np.asarray(alpha), np.ones((3, 3), np.uint8), iterations=1)
    alpha = Image.fromarray(alpha_np).filter(ImageFilter.GaussianBlur(radius=3))
    base.paste(person, (left, top), alpha)
