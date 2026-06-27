"""FastMCP server exposing melt-7 (MLT) for agentic video editing/composition.

Thin tool-definition layer: all MLT XML mutation lives in mlt_project.py,
all subprocess/melt-7/ffprobe work lives in melt_client.py. Every tool here
either returns plain data or raises -- FastMCP turns an uncaught exception
into an MCP tool error automatically.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

import melt_client
import mlt_project

mcp = FastMCP("melt-7")

_GENERATOR_PREFIXES = {"color:": "color", "noise:": "noise"}


def _resolve_source(source: str) -> tuple[str, str, bool]:
    """Returns (mlt_service, resource, is_generator) for an add_clip 'source' string."""
    for prefix, service in _GENERATOR_PREFIXES.items():
        if source.startswith(prefix):
            return service, source[len(prefix):], True
    return "avformat", source, False


# ---------------------------------------------------------------------------
# Project lifecycle
# ---------------------------------------------------------------------------

@mcp.tool()
def create_project(name: str, profile: str = "atsc_1080p_25") -> dict:
    """Create a new melt-7 video editing project.

    Args:
        name: Human-readable project name (does not need to be unique).
        profile: MLT profile id controlling resolution/fps/aspect. See
            query_services(kind="profiles") for the full list, e.g.
            "atsc_1080p_30", "dv_pal", "hdv_720_25".

    Returns:
        dict with 'project_id' (use this in all other tool calls), 'name', 'profile'.
    """
    profile_attrs = melt_client.query("profiles", service_id=profile)
    return mlt_project.create_project(name, profile, profile_attrs)


@mcp.tool()
def list_projects() -> dict:
    """List all existing projects with their id, name, profile, and track/clip counts.

    Returns:
        dict with 'projects': list of {project_id, name, profile, created_at, num_tracks, num_clips}.
    """
    return {"projects": mlt_project.list_projects()}


@mcp.tool()
def get_project_xml(project_id: str) -> dict:
    """Return the full current MLT XML for a project, plus a human-readable summary.

    Args:
        project_id: The project to inspect.

    Returns:
        dict with 'xml' (raw MLT XML string) and 'summary' (tracks with their clips,
        in/out points in seconds, and any filters/transitions attached).
    """
    root = mlt_project.load_tree(project_id).getroot()
    return {"xml": mlt_project.to_xml_string(project_id), "summary": mlt_project.summarize(root)}


@mcp.tool()
def delete_project(project_id: str) -> dict:
    """Permanently delete a project's XML and all of its rendered outputs.

    Args:
        project_id: The project to delete.

    Returns:
        dict with 'deleted': true/false (false if the project did not exist).
    """
    return {"deleted": mlt_project.delete_project(project_id)}


# ---------------------------------------------------------------------------
# Source inspection
# ---------------------------------------------------------------------------

@mcp.tool()
def probe_clip(file_path: str) -> dict:
    """Inspect a media file with ffprobe to get duration, resolution, fps, and codecs
    before placing it in a project (melt-7 itself has no simple probe mode).

    Args:
        file_path: Absolute path to a video, audio, or image file.

    Returns:
        dict with 'duration_seconds', 'width', 'height', 'fps', 'video_codec',
        'audio_codec', 'has_video', 'has_audio'. Fields are null where not
        applicable (e.g. an audio-only file has width/height = null).
    """
    return melt_client.probe(file_path)


# ---------------------------------------------------------------------------
# Timeline structure
# ---------------------------------------------------------------------------

@mcp.tool()
def add_track(project_id: str, kind: str = "video", position: int | None = None) -> dict:
    """Add a new track (playlist) to the project's timeline.

    Args:
        project_id: The project to modify.
        kind: "video" or "audio" -- purely informational/organizational; melt-7
            treats all tracks uniformly, but this is recorded so get_project_xml's
            summary can label tracks sensibly.
        position: 0-based index to insert at (0 = bottom of the stack). Defaults
            to appending as the new topmost track.

    Returns:
        dict with 'track_id' (e.g. "track2") and 'position'.
    """
    return mlt_project.add_track(project_id, kind, position)


@mcp.tool()
def remove_track(project_id: str, track_id: str) -> dict:
    """Remove an entire track and any transitions that reference it (transition
    track indices on other tracks are automatically renumbered).

    Args:
        project_id: The project to modify.
        track_id: Track to remove, e.g. "track1".

    Returns:
        dict with 'removed': true, 'track_id'.
    """
    mlt_project.remove_track(project_id, track_id)
    return {"removed": True, "track_id": track_id}


@mcp.tool()
def add_clip(
    project_id: str,
    track_id: str,
    source: str,
    clip_in: float = 0.0,
    clip_out: float | None = None,
    position: float | None = None,
) -> dict:
    """Place a clip on a track, sourced from a media file or an MLT generator.

    Args:
        project_id: The project to modify.
        track_id: Target track (from add_track or get_project_xml), e.g. "track0".
        source: Absolute path to a media file (e.g. "/path/clip.mp4"), OR an MLT
            generator spec such as "color:red", "color:0xFF0000FF" (8-hex-digit
            RGBA), or "noise:". Use probe_clip first on real media files if you
            need to know their actual duration before choosing clip_in/clip_out.
        clip_in: Start point within the source, in seconds (default 0.0).
        clip_out: End point within the source, in seconds. Required for generator
            sources (which have no inherent duration); for media files, defaults
            to the source's full probed duration if omitted.
        position: Timeline position (seconds from track start) to place the clip
            at. Defaults to appending immediately after the last clip/blank on
            the track. If position is later than the current end of the track,
            a blank gap is inserted automatically. Positions earlier than the
            current end (i.e. overlapping existing content) are rejected --
            use move_clip/trim_clip/remove_clip to rearrange instead.

    Returns:
        dict with 'clip_index' (0-based position of the clip within the track --
        use this to address it in trim_clip/move_clip/remove_clip), 'track_id',
        'timeline_start_seconds', 'timeline_end_seconds'.
    """
    service, resource, is_generator = _resolve_source(source)
    if clip_out is None:
        if is_generator:
            raise ValueError("clip_out is required for generator sources (color:, noise:)")
        info = melt_client.probe(resource)
        if not info.get("duration_seconds"):
            raise ValueError(f"could not determine duration of {resource!r} via ffprobe")
        clip_out = info["duration_seconds"]

    return mlt_project.add_clip(
        project_id, track_id,
        producer_service=service, producer_resource=resource,
        clip_in_seconds=clip_in, clip_out_seconds=clip_out,
        position_seconds=position,
    )


@mcp.tool()
def trim_clip(
    project_id: str,
    track_id: str,
    clip_index: int,
    clip_in: float | None = None,
    clip_out: float | None = None,
) -> dict:
    """Change the in/out trim points of an already-placed clip, without moving
    its position on the timeline.

    Args:
        project_id: The project to modify.
        track_id: Track the clip lives on.
        clip_index: Index of the clip within the track (from add_clip or get_project_xml).
        clip_in: New in-point in seconds within the source (omit to leave unchanged).
        clip_out: New out-point in seconds within the source (omit to leave unchanged).

    Returns:
        dict with updated 'clip_in_seconds', 'clip_out_seconds', 'duration_seconds'.
    """
    return mlt_project.trim_clip(project_id, track_id, clip_index, clip_in, clip_out)


@mcp.tool()
def move_clip(
    project_id: str,
    track_id: str,
    clip_index: int,
    new_position: float,
    new_track_id: str | None = None,
) -> dict:
    """Move a clip to a new timeline position, optionally onto a different track.
    Leaves a blank gap where the clip used to be.

    Args:
        project_id: The project to modify.
        track_id: Track the clip currently lives on.
        clip_index: Index of the clip to move.
        new_position: New start time on the timeline, in seconds.
        new_track_id: If set, moves the clip to this track instead of track_id.

    Returns:
        dict with 'track_id' (final track), 'clip_index' (new index on that track),
        'timeline_start_seconds'.
    """
    return mlt_project.move_clip(project_id, track_id, clip_index, new_position, new_track_id)


@mcp.tool()
def remove_clip(project_id: str, track_id: str, clip_index: int) -> dict:
    """Remove a clip from a track, replacing it with a blank gap of the same
    length so clips on other tracks that were time-aligned with this one don't
    shift out of sync. To close the gap, use move_clip on subsequent clips.

    Args:
        project_id: The project to modify.
        track_id: Track the clip lives on.
        clip_index: Index of the clip to remove.

    Returns:
        dict with 'removed': true, 'track_id', 'clip_index'.
    """
    mlt_project.remove_clip(project_id, track_id, clip_index)
    return {"removed": True, "track_id": track_id, "clip_index": clip_index}


# ---------------------------------------------------------------------------
# Transitions and filters
# ---------------------------------------------------------------------------

@mcp.tool()
def add_transition(
    project_id: str,
    track_a: str,
    track_b: str,
    service: str = "qtblend",
    properties: dict[str, str] | None = None,
) -> dict:
    """Add a transition (composite/wipe/mix) between two tracks on the master tractor.

    Args:
        project_id: The project to modify.
        track_a: The lower/background track id (e.g. "track0").
        track_b: The upper/foreground track id (e.g. "track1").
        service: MLT transition service name, e.g. "qtblend" (pan/zoom/rotate
            compositing, the default for video-over-video), "luma" (wipe/dissolve),
            "mix" (audio crossfade), "composite" (legacy alpha compositing). Use
            query_services(kind="transitions") to discover all available services,
            and query_services(kind="transitions", service_id=<name>) for that
            service's full parameter list.
        properties: Extra MLT properties for the transition, e.g. {"softness": "0.5"}
            for a luma wipe, or {"always_active": "1"} to apply across the whole
            timeline rather than only where both tracks have clips.

    Returns:
        dict with 'transition_id'.
    """
    return {"transition_id": mlt_project.add_transition(project_id, track_a, track_b, service, properties)}


@mcp.tool()
def add_filter(
    project_id: str,
    target: str,
    service: str,
    properties: dict[str, str] | None = None,
    clip_index: int | None = None,
) -> dict:
    """Attach an MLT filter to a track, a specific clip on a track, or the whole project.

    Args:
        project_id: The project to modify.
        target: "project" to filter the final master-tractor output, "track:<id>"
            to filter a whole track (e.g. "track:track0"), or a bare track id
            (e.g. "track0") combined with clip_index to filter one clip.
        service: MLT filter service name, e.g. "brightness", "volume", "sepia",
            "frei0r.cairoblend". Use query_services(kind="filters") to discover
            available services, and query_services(kind="filters", service_id=<name>)
            for parameters.
        properties: Filter properties as MLT property name/value strings, e.g.
            {"level": "0.75"} for brightness, {"gain": "2.0"} for volume.
        clip_index: Required when target is a bare track id, to select which clip
            on that track to filter.

    Returns:
        dict with 'filter_id'.
    """
    return {"filter_id": mlt_project.add_filter(project_id, target, service, properties, clip_index)}


@mcp.tool()
def remove_filter(project_id: str, filter_id: str) -> dict:
    """Remove a previously added filter by its id (from add_filter's return value
    or from get_project_xml's summary).

    Args:
        project_id: The project to modify.
        filter_id: The filter to remove, e.g. "filter0".

    Returns:
        dict with 'removed': true.
    """
    mlt_project.remove_filter(project_id, filter_id)
    return {"removed": True}


@mcp.tool()
def add_text_overlay(
    project_id: str,
    track_id: str,
    text: str,
    position: float | None = None,
    duration: float = 5.0,
    font_size: int = 48,
    fgcolour: str = "0xffffffff",
    bgcolour: str = "0x00000000",
    halign: str = "center",
) -> dict:
    """Add a text/title clip to a track using MLT's qtext producer.

    Args:
        project_id: The project to modify.
        track_id: Track to place the text on (typically a dedicated overlay
            track composited over video tracks below it via add_transition).
        text: The text to render.
        position: Timeline position in seconds to start the overlay (defaults
            to appending after the last clip on this track).
        duration: How long the text is shown, in seconds.
        font_size: Point size of the rendered text.
        fgcolour: Text color as 0xRRGGBBAA hex (default opaque white).
        bgcolour: Background color as 0xRRGGBBAA hex (default fully transparent).
        halign: Paragraph alignment: "left", "center", or "right".

    Returns:
        dict with 'clip_index', 'track_id', 'timeline_start_seconds', 'timeline_end_seconds'.
        Note: for the text to be visible over other tracks, ensure this track has
        a transition (e.g. qtblend) compositing it over the tracks below --
        call add_transition if one isn't already present.
    """
    extra_props = {
        "text": text,
        "fgcolour": fgcolour,
        "bgcolour": bgcolour,
        "size": str(font_size),
        "align": halign,
    }
    # qtext renders from the "text" property, not "resource" (confirmed empirically) --
    # resource is only given a unique throwaway value so distinct overlays never
    # collide with each other via producer deduplication.
    unique_resource = f"+{uuid.uuid4().hex}"
    return mlt_project.add_clip(
        project_id, track_id,
        producer_service="qtext", producer_resource=unique_resource,
        clip_in_seconds=0.0, clip_out_seconds=duration,
        position_seconds=position,
        producer_extra_props=extra_props,
    )


# ---------------------------------------------------------------------------
# Discovery / escape hatch
# ---------------------------------------------------------------------------

@mcp.tool()
def query_services(kind: str, service_id: str | None = None) -> dict:
    """Discover what MLT producers, filters, transitions, consumers, or profiles
    are available on this system, wrapping `melt-7 -query`, rather than relying
    on a hardcoded list (the installed MLT build's services can vary by system).

    Args:
        kind: One of "producers", "filters", "transitions", "consumers", "profiles",
            "formats", "video_codecs", "audio_codecs".
        service_id: If given, return full parameter/schema details for this one
            service/profile id instead of the full list (only supported for
            "producers", "filters", "transitions", "consumers", "profiles" --
            e.g. kind="filters", service_id="brightness").

    Returns:
        dict with 'kind' and either 'services' (list of ids) or 'detail' (a dict
        of schema info: identifier, title, description, parameters: [...]).
    """
    result = melt_client.query(kind, service_id)
    if service_id is not None:
        return {"kind": kind, "detail": result}
    return {"kind": kind, "services": result}


@mcp.tool()
def set_raw_xml_property(project_id: str, element_id: str, name: str, value: str) -> dict:
    """Escape hatch: set an arbitrary MLT property by name on any existing element
    (producer, playlist, tractor, filter, transition) identified by its id
    attribute. For cases the high-level tools don't cover (obscure filter
    parameters, custom metadata, etc).

    Args:
        project_id: The project to modify.
        element_id: The id="..." of the target element (from get_project_xml),
            or "mlt" to target the document root.
        name: MLT property name to set, e.g. "mlt_service" or a custom namespaced name.
        value: String value to set.

    Returns:
        dict with 'element_id', 'name', 'value'.
    """
    mlt_project.set_raw_xml_property(project_id, element_id, name, value)
    return {"element_id": element_id, "name": name, "value": value}


@mcp.tool()
def remove_raw_xml_property(project_id: str, element_id: str, name: str) -> dict:
    """Remove a property by name from an element. Counterpart to set_raw_xml_property.

    Args:
        project_id: The project to modify.
        element_id: The id="..." of the target element, or "mlt" for the document root.
        name: Property name to remove.

    Returns:
        dict with 'removed': true/false (false if no such property existed).
    """
    removed = mlt_project.remove_raw_xml_property(project_id, element_id, name)
    return {"removed": removed, "element_id": element_id, "name": name}


@mcp.tool()
def inject_raw_xml(project_id: str, parent_id: str, xml_fragment: str, index: int | None = None) -> dict:
    """Escape hatch: parse an arbitrary XML fragment (e.g. a full <filter>...</filter>
    or <transition>...</transition> block) and insert it as a child of an existing
    element, for structures too complex or obscure for add_filter/add_transition.

    Args:
        project_id: The project to modify.
        parent_id: id="..." of the element to insert the fragment under (e.g. a
            track's playlist id, or "mlt" for the document root), or "mlt" for
            the document root.
        xml_fragment: A single well-formed XML element as a string, e.g.
            '<filter><property name="mlt_service">grain</property></filter>'.
            If it has no id attribute, one is assigned automatically.
        index: Position among parent_id's children to insert at (default: append).

    Returns:
        dict with 'element_id' (the id of the inserted element, possibly
        auto-assigned), 'parent_id'.
    """
    element_id = mlt_project.inject_raw_xml(project_id, parent_id, xml_fragment, index)
    return {"element_id": element_id, "parent_id": parent_id}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

@mcp.tool()
def render_project(
    project_id: str,
    output_name: str | None = None,
    vcodec: str = "libx264",
    acodec: str = "aac",
    extra_args: dict[str, str] | None = None,
    timeout_seconds: int = 600,
) -> dict:
    """Render the project's current timeline to an MP4 file via melt-7. This is a
    synchronous call -- it blocks until the render finishes or times out.

    Args:
        project_id: The project to render.
        output_name: Filename for the rendered output (default: timestamped, e.g.
            "render_20260623_171530.mp4"). Always written under this project's
            renders/ directory regardless of any path components given.
        vcodec: Video codec for the avformat consumer (default "libx264").
        acodec: Audio codec for the avformat consumer (default "aac").
        extra_args: Additional consumer name=value properties to pass through
            verbatim, e.g. {"b:v": "4M"} (see query_services(kind="consumers",
            service_id="avformat") for the full option list).
        timeout_seconds: Kill the render and report failure if it exceeds this
            (default 600s / 10 minutes).

    Returns:
        dict with:
          'success': bool -- True only if the output file exists, has nonzero
                      size, ffprobe can read a valid duration from it, that
                      duration is not implausibly shorter than the project's
                      expected duration, AND stderr/stdout contain no known
                      failure markers (e.g. "failed to load producer") (melt-7's
                      exit code alone is not a reliable success signal --
                      confirmed it can exit 0 while still logging a load
                      failure and silently substituting a blank for the clip).
          'output_path': absolute path to the rendered file (present even on
                      partial failure, if melt-7 wrote anything).
          'duration_seconds', 'width', 'height': from ffprobe on the result.
          'stdout', 'stderr': full captured subprocess output, for debugging a
                      broken composition.
          'warnings': suspicious stderr lines detected (e.g. "failed to load"),
                      surfaced even when other checks pass.
          'command': the exact argv list invoked, for transparency/debuggability.
    """
    root = mlt_project.load_tree(project_id).getroot()
    fps = mlt_project.get_profile_fps(root)
    expected_duration = mlt_project.frames_to_seconds(mlt_project.project_duration_frames(root), fps)

    if output_name:
        output_name = Path(output_name).name
    else:
        output_name = f"render_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    if not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"

    output_path = mlt_project.renders_dir(project_id) / output_name
    return melt_client.render(
        mlt_project.project_xml_path(project_id), output_path,
        vcodec, acodec, extra_args, timeout_seconds, expected_duration,
    )


CORS = [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])]

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001, middleware=CORS, stateless_http=True)
