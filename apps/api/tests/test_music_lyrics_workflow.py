"""Music Lyrics Workflow v2.0 at the API (client specification, 9 Sep 2026).

Two halves: the request controls the workflow adds — gated by
`settings.lyrics_workflow`, same policy as lyrics — and the `result` a
worker attaches on completion, which is how the lyrics, their timing and the
coverage numbers reach the customer.
"""

from __future__ import annotations

import json

from httpx import AsyncClient

from tests.test_worker_protocol import claim, register


def _music(parameters: dict) -> dict:
    return {
        "workflow_id": "music",
        "prompt": "an upbeat pop song about summer in Lahore, hopeful, female vocals",
        "parameters": {"duration": "1m", **parameters},
    }


async def test_music_accepts_the_lyrics_workflow_controls(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/generations",
        json=_music(
            {
                "rhyme_scheme": "ABAB",
                "rhyme_mode": "relaxed",
                "point_of_view": "first_person",
                "clean_mode": False,
                "reference_audio_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "dry_run": True,
            }
        ),
    )
    assert response.status_code == 202, response.text
    job = (await client.get(f"/api/v1/generations/{response.json()['job_id']}")).json()
    stored = job["parameters"]
    assert stored["rhyme_scheme"] == "ABAB"
    assert stored["rhyme_mode"] == "relaxed"
    assert stored["point_of_view"] == "first_person"
    assert stored["clean_mode"] is False
    assert stored["reference_audio_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert stored["dry_run"] is True
    # Nothing reported yet.
    assert job["result"] is None


async def test_the_controls_are_refused_where_the_workflow_is_not_declared(
    client: AsyncClient, text_to_video_request: dict
) -> None:
    """Same policy as lyrics: present-and-unsupported is reported, never
    silently dropped."""
    request = json.loads(json.dumps(text_to_video_request))
    request["parameters"].update(
        {"rhyme_scheme": "AABB", "clean_mode": True, "dry_run": True, "reference_audio_url": "https://youtu.be/x"}
    )
    response = await client.post("/api/v1/generations", json=request)
    assert response.status_code == 422
    fields = {f["field"] for f in response.json()["error"]["details"]["fields"]}
    assert {"rhyme_scheme", "clean_mode", "dry_run", "reference_audio_url"} <= fields


async def test_an_unknown_rhyme_scheme_is_a_schema_error(client: AsyncClient) -> None:
    response = await client.post("/api/v1/generations", json=_music({"rhyme_scheme": "ABBA"}))
    assert response.status_code == 422


async def test_a_reference_link_must_be_on_a_host_the_worker_fetches(client: AsyncClient) -> None:
    for bad in (
        "http://www.youtube.com/watch?v=abc",
        "https://drive.google.com/file/d/abc",
        "https://user:pw@soundcloud.com/x/y",
        "ftp://vimeo.com/1",
    ):
        response = await client.post("/api/v1/generations", json=_music({"reference_audio_url": bad}))
        assert response.status_code == 422, bad
        fields = {f["field"] for f in response.json()["error"]["details"]["fields"]}
        assert "reference_audio_url" in fields, bad

    ok = await client.post(
        "/api/v1/generations", json=_music({"reference_audio_url": "https://soundcloud.com/artist/track"})
    )
    assert ok.status_code == 202, ok.text


async def test_music_offers_an_optional_reference_upload(client: AsyncClient) -> None:
    workflow = (await client.get("/api/v1/workflows/music")).json()
    assert workflow["settings"]["lyrics_workflow"] is True
    roles = {item["role"]: item for item in workflow["inputs"]}
    assert roles["reference_audio"]["kind"] == "audio"
    assert roles["reference_audio"]["required"] is False
    # And no reference is needed to make a song.
    response = await client.post("/api/v1/generations", json=_music({}))
    assert response.status_code == 202, response.text


async def test_a_completion_may_carry_a_result_that_reaches_the_customer(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)

    result = {
        "workflow": "music-lyrics/2.0",
        "lyrics": "[verse]\nline one\nline two",
        "lyrics_lrc": "[00:00.00]line one\n[00:04.00]line two\n",
        "planned_vocal_coverage": 0.93,
        "measured_vocal_coverage": 0.91,
        "rhyme_pass_rate": 1.0,
        "warnings": [],
    }
    ack = await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/complete",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "output_key": job["output_upload_key"],
            "output_kind": "image",
            "output_content_type": "image/png",
            "size_bytes": 4096,
            "result": result,
        },
    )
    assert ack.json()["accepted"] is True

    public = (await client.get(f"/api/v1/generations/{job['job_id']}")).json()
    assert public["status"] == "completed"
    assert public["result"] == result

    # And in the listing.
    listed = (await client.get("/api/v1/generations")).json()
    mine = next(item for item in listed["items"] if item["id"] == job["job_id"])
    assert mine["result"]["measured_vocal_coverage"] == 0.91


async def test_a_completion_without_a_result_is_byte_for_byte_the_old_contract(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)
    ack = await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/complete",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "output_key": job["output_upload_key"],
            "output_kind": "image",
            "output_content_type": "image/png",
            "size_bytes": 4096,
        },
    )
    assert ack.json()["accepted"] is True
    public = (await client.get(f"/api/v1/generations/{job['job_id']}")).json()
    assert public["result"] is None


async def test_an_oversized_result_is_refused(
    client: AsyncClient, worker_headers: dict, text_to_video_request: dict
) -> None:
    await client.post("/api/v1/generations", json=text_to_video_request)
    worker_id = await register(client, worker_headers)
    job = await claim(client, worker_headers, worker_id)
    response = await client.post(
        f"/api/v1/internal/jobs/{job['job_id']}/complete",
        headers=worker_headers,
        json={
            "worker_id": worker_id,
            "lease_token": job["lease_token"],
            "output_key": job["output_upload_key"],
            "output_kind": "image",
            "output_content_type": "image/png",
            "size_bytes": 4096,
            "result": {"lyrics": "x" * (70 * 1024)},
        },
    )
    assert response.status_code == 422
