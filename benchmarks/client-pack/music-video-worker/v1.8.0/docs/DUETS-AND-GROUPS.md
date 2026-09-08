# Duets and groups

The request supports one to five supplied performers. Each performer may have up to four identity-reference images, allowing a maximum of 20 reference images across a five-person group.

## Automatic scene coverage

When the request JSON is created without `max_people_visible_per_shot`, the worker automatically uses the number of supplied performers, capped at five. A five-member band therefore defaults to five people in eligible group scenes.

The planner does not force all members into every scene. It rotates individual coverage across the supplied performer IDs and adds selected ensemble scenes in medium performance, wide movement and climax families. Wide performance scenes include the complete permitted group. This provides readable solo coverage while still making the result feel like a band video.

Set a smaller explicit limit when the video model or GPU configuration handles fewer simultaneous identities more reliably:

```json
{
  "max_people_visible_per_shot": 3
}
```

## Five-member request

```json
{
  "job_id": "band-five-0001",
  "audio": "/srv/zolexai/uploads/band-song.wav",
  "prompt": "make me a cinematic video for this five-person band according to the lyrics",
  "performers": [
    {
      "id": "lead-vocalist",
      "role": "lead_vocalist",
      "reference_images": ["/srv/zolexai/uploads/lead-front.jpg"]
    },
    {
      "id": "guitarist",
      "role": "guitarist",
      "reference_images": ["/srv/zolexai/uploads/guitarist-front.jpg"]
    },
    {
      "id": "bassist",
      "role": "bassist",
      "reference_images": ["/srv/zolexai/uploads/bassist-front.jpg"]
    },
    {
      "id": "keyboardist",
      "role": "keyboardist",
      "reference_images": ["/srv/zolexai/uploads/keyboardist-front.jpg"]
    },
    {
      "id": "drummer",
      "role": "drummer",
      "reference_images": ["/srv/zolexai/uploads/drummer-front.jpg"]
    }
  ],
  "max_people_visible_per_shot": 5,
  "formats": ["16:9"]
}
```

Roles and descriptions are inserted into both anchor and video prompts. Recognized instrumental roles—including drummer, percussionist, guitarist, bassist, keyboardist, pianist and DJ—receive role-appropriate solo actions instead of being incorrectly told to sing.

## Identity rules

Every group prompt identifies the assigned people separately and instructs the image/video backend to preserve each person's face, hair, proportions, wardrobe, tattoos, jewelry and accessories. It explicitly forbids blending, duplicating or swapping faces, bodies, clothing or instruments and forbids unassigned people.

Multi-person identity accuracy still depends on the production anchor generator and renderer. For best results, provide at least one clear front-facing image per member; up to four angles per person are accepted. A group photograph can supplement individual references, but separate identity images are strongly recommended so the backend can distinguish all five people.
