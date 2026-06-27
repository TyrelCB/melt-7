# melt-7 MCP Server

An MCP server exposing [melt-7](https://www.mltframework.org/) (the MLT
Multimedia Framework's CLI rendering engine -- the same backend Kdenlive
uses) for agentic video editing/composition: build a multi-track timeline
with clips, transitions, filters, and text overlays, then render it to MP4.

## Requirements

- `melt-7` and `ffprobe`/`ffmpeg` on `PATH` (Fedora: `dnf install mlt`)
- Python 3.10+

```bash
pip install -r requirements.txt
```

## Running

```bash
python mcp_server.py
```

Serves MCP over streamable HTTP on `http://0.0.0.0:8001/mcp` (port 8001, to
avoid colliding with the wan2_2_t2v server on 8000 if both run on the same host).

## Project model

Each call to `create_project` creates a project directory under `projects/<id>/`
holding `project.mlt` (the authoritative MLT XML, hand-rolled to a flat schema
rather than Kdenlive's nested per-track-tractor convention), `meta.json` (display
name/profile bookkeeping), and a `renders/` directory for rendered output. Every
tool re-reads `project.mlt` fresh and atomically rewrites it, so state survives
server restarts -- there is no in-memory project cache.

## Tools

| Tool | Purpose |
|---|---|
| `create_project(name, profile)` | Start a new project at a given MLT profile (resolution/fps) |
| `list_projects()` | List all projects with track/clip counts |
| `get_project_xml(project_id)` | Raw MLT XML + human-readable summary |
| `delete_project(project_id)` | Delete a project and its renders |
| `probe_clip(file_path)` | ffprobe a media file (duration, resolution, fps, codecs) |
| `add_track(project_id, kind, position)` | Add a track to the timeline |
| `remove_track(project_id, track_id)` | Remove a track |
| `add_clip(project_id, track_id, source, clip_in, clip_out, position)` | Place a clip (file path or `color:`/`noise:` generator) |
| `trim_clip(project_id, track_id, clip_index, clip_in, clip_out)` | Change a clip's in/out points |
| `move_clip(project_id, track_id, clip_index, new_position, new_track_id)` | Move a clip (leaves a blank gap behind) |
| `remove_clip(project_id, track_id, clip_index)` | Remove a clip (leaves a blank gap) |
| `add_transition(project_id, track_a, track_b, service, properties)` | Composite/wipe/mix between two tracks |
| `add_filter(project_id, target, service, properties, clip_index)` | Attach a filter to a track, clip, or the whole project |
| `remove_filter(project_id, filter_id)` | Remove a filter |
| `add_text_overlay(project_id, track_id, text, ...)` | Convenience wrapper for a `qtext` title clip |
| `query_services(kind, service_id)` | Discover available producers/filters/transitions/consumers/profiles |
| `set_raw_xml_property` / `remove_raw_xml_property` / `inject_raw_xml` | Escape hatch for anything the above don't cover |
| `render_project(project_id, output_name, vcodec, acodec, extra_args, timeout_seconds)` | Render the timeline to MP4 (synchronous) |

## Notes on reliability

melt-7's process exit code is **not** a reliable success signal -- it can
exit 0 while logging a "failed to load producer" error and silently
substituting a blank clip. `render_project` verifies success independently:
the output file must exist, be ffprobe-readable, have a duration close to
the project's expected duration, and stdout/stderr must contain no known
failure markers.
