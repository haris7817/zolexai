from __future__ import annotations

from .config import WorkerConfig
from .models import MusicVideoRequest
from .utils import read_json, validate_job_id
from .worker import run_job

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install the API extra: pip install '.[api]'") from exc


app = FastAPI(title="ZolexAI Music Video Worker", version="1.8.0")


def _config() -> WorkerConfig:
    return WorkerConfig.from_env()


def _run_background(request: MusicVideoRequest) -> None:
    run_job(request, _config())


@app.get("/healthz")
def health() -> dict:
    config = _config()
    return {
        "ok": True,
        "fps": config.fps,
        "anchor_backend": config.anchor_backend,
        "render_backend": config.render_backend,
        "render_concurrency": config.render_concurrency,
        "upscale_backend": config.upscale_backend,
        "target_job_seconds": config.target_job_seconds,
        "transcription_backend": config.transcription_backend,
        "reference_video_analysis": "built_in_metrics_plus_optional_ai_vision_command",
        "reference_video_url_input": True,
        "scene_delivery": {"landscape": "1280x704", "portrait": "704x1280"},
        "final_delivery": {"landscape": "3840x2160", "portrait": "2160x3840"},
    }


@app.post("/v1/music-video-jobs", status_code=202)
def create_job(payload: dict, background_tasks: BackgroundTasks) -> dict:
    try:
        request = MusicVideoRequest.from_dict(payload)
        config = _config()
        config.validate()
        background_tasks.add_task(_run_background, request)
        return {
            "job_id": request.job_id,
            "status": "accepted",
            "status_path": f"/v1/music-video-jobs/{request.job_id}",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/music-video-jobs/{job_id}")
def job_status(job_id: str) -> dict:
    try:
        job_id = validate_job_id(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = _config().work_root / job_id / "status.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown job")
    return read_json(path)


@app.post("/v1/music-video-jobs/{job_id}/cancel", status_code=202)
def cancel_job(job_id: str) -> dict:
    try:
        job_id = validate_job_id(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_dir = _config().work_root / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Unknown job")
    (job_dir / "CANCEL").write_text("cancel requested\n", encoding="utf-8")
    return {"job_id": job_id, "cancel_requested": True}
