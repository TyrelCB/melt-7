"""MLT XML data model: load/mutate/save .mlt project files via ElementTree.

The on-disk project.mlt file is always the source of truth. Every mutating
function here loads it fresh, mutates the tree, and atomically rewrites it
before returning -- there is no in-memory project cache, so state survives
server restarts and concurrent access to different projects is trivially
safe.

Schema generated/consumed (deliberately flatter than real Kdenlive XML,
which nests a tractor per UI track for reasons we don't need):

  <mlt root="...">
    <profile .../>
    <producer id="producer0" in=".." out="..">         one per distinct source
      <property name="resource">...</property>
      <property name="mlt_service">avformat|color|qtext|noise</property>
    </producer>
    <playlist id="track0">                              one per agent-visible track
      <entry producer="producer0" in=".." out="..">
        <filter id="filter0">...</filter>                optional, scoped to this clip
      </entry>
      <blank length=".."/>
    </playlist>
    <tractor id="main_tractor">                          exactly one; track order = stack order
      <track producer="track0"/>
      <transition id="transition0">
        <property name="mlt_service">qtblend</property>
        <property name="a_track">0</property>
        <property name="b_track">1</property>
      </transition>
      <filter id="filter1">...</filter>                  optional, scoped to whole output
    </tractor>
  </mlt>

Producers always use the real (non-"novalidate") avformat service for real
media files: confirmed empirically that avformat-novalidate silently
truncates the render to near-zero length on a missing/unreadable source
with no warning, whereas plain avformat substitutes a blank and logs a
clear "failed to load chain" message while preserving the full expected
duration -- much safer for a server that needs reliable error signals.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

PROJECTS_DIR = Path(__file__).parent / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)

MLT_VERSION = "7.32.0"
KIND_PROPERTY = "melt7mcp:kind"


class ProjectNotFoundError(LookupError):
    pass


class TrackNotFoundError(LookupError):
    pass


class ClipNotFoundError(LookupError):
    pass


class ElementNotFoundError(LookupError):
    pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def project_xml_path(project_id: str) -> Path:
    return project_dir(project_id) / "project.mlt"


def meta_path(project_id: str) -> Path:
    return project_dir(project_id) / "meta.json"


def renders_dir(project_id: str) -> Path:
    return project_dir(project_id) / "renders"


def project_exists(project_id: str) -> bool:
    return project_xml_path(project_id).exists()


def _require_project(project_id: str) -> None:
    if not project_exists(project_id):
        raise ProjectNotFoundError(f"no project with id {project_id!r}")


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_tree(project_id: str) -> ET.ElementTree:
    _require_project(project_id)
    return ET.parse(project_xml_path(project_id))


def _reorder_root_children(root: ET.Element) -> None:
    """melt-7's XML parser resolves id references (e.g. a tractor's <track
    producer="trackN"/>) in document order, so any element must be defined
    before something else references its id. Confirmed empirically: a
    <tractor> written before the <playlist>/<producer> elements it points to
    silently fails to pick up their full length. We always append new
    elements wherever is convenient when mutating, then restore the required
    order (profile, then producers, then playlists, then the tractor last)
    here, in the one place all mutations funnel through before being written.
    """
    producers = [el for el in root if el.tag == "producer"]
    playlists = [el for el in root if el.tag == "playlist"]
    tractors = [el for el in root if el.tag == "tractor"]
    others = [el for el in root if el.tag not in ("producer", "playlist", "tractor")]
    for el in list(root):
        root.remove(el)
    for el in others + producers + playlists + tractors:
        root.append(el)


def save_tree(project_id: str, tree: ET.ElementTree) -> None:
    _reorder_root_children(tree.getroot())
    ET.indent(tree, space="  ")
    path = project_xml_path(project_id)
    tmp_path = path.with_suffix(".tmp")
    tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
    os.replace(tmp_path, path)


def to_xml_string(project_id: str) -> str:
    tree = load_tree(project_id)
    ET.indent(tree, space="  ")
    body = ET.tostring(tree.getroot(), encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + body


def load_meta(project_id: str) -> dict:
    if not meta_path(project_id).exists():
        return {}
    return json.loads(meta_path(project_id).read_text())


def save_meta(project_id: str, meta: dict) -> None:
    meta_path(project_id).write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Generic element/property helpers (also used by the raw-XML escape hatch)
# ---------------------------------------------------------------------------

def get_property(element: ET.Element, name: str) -> str | None:
    for prop in element.findall("property"):
        if prop.get("name") == name:
            return prop.text
    return None


def set_property(element: ET.Element, name: str, value: str) -> None:
    for prop in element.findall("property"):
        if prop.get("name") == name:
            prop.text = value
            return
    prop = ET.SubElement(element, "property", {"name": name})
    prop.text = value


def remove_property(element: ET.Element, name: str) -> bool:
    for prop in element.findall("property"):
        if prop.get("name") == name:
            element.remove(prop)
            return True
    return False


def find_by_id(root: ET.Element, element_id: str) -> ET.Element:
    for el in root.iter():
        if el.get("id") == element_id:
            return el
    raise ElementNotFoundError(f"no element with id {element_id!r}")


def find_parent(root: ET.Element, target: ET.Element) -> ET.Element | None:
    for el in root.iter():
        if target in list(el):
            return el
    return None


def new_id(root: ET.Element, prefix: str) -> str:
    nums = []
    for el in root.iter():
        eid = el.get("id")
        if eid and eid.startswith(prefix) and eid[len(prefix):].isdigit():
            nums.append(int(eid[len(prefix):]))
    return f"{prefix}{max(nums) + 1 if nums else 0}"


# ---------------------------------------------------------------------------
# Profile / time helpers
# ---------------------------------------------------------------------------

def get_profile_fps(root: ET.Element) -> float:
    profile = root.find("profile")
    num = float(profile.get("frame_rate_num", 25))
    den = float(profile.get("frame_rate_den", 1)) or 1.0
    return num / den


def seconds_to_frames(seconds: float, fps: float) -> int:
    return max(0, round(seconds * fps))


def frames_to_seconds(frames: int, fps: float) -> float:
    return frames / fps


# ---------------------------------------------------------------------------
# Track (playlist) helpers
# ---------------------------------------------------------------------------

def _timeline_children(playlist: ET.Element) -> list[ET.Element]:
    return [el for el in playlist if el.tag in ("entry", "blank")]


def playlist_length_frames(playlist: ET.Element) -> int:
    total = 0
    for el in _timeline_children(playlist):
        if el.tag == "entry":
            total += int(el.get("out")) - int(el.get("in")) + 1
        else:
            total += int(el.get("length"))
    return total


def find_track(root: ET.Element, track_id: str) -> ET.Element:
    for playlist in root.findall("playlist"):
        if playlist.get("id") == track_id:
            return playlist
    raise TrackNotFoundError(f"no track {track_id!r}")


def list_track_ids(root: ET.Element) -> list[str]:
    tractor = root.find("tractor")
    return [tr.get("producer") for tr in tractor.findall("track")]


def project_duration_frames(root: ET.Element) -> int:
    lengths = [playlist_length_frames(find_track(root, tid)) for tid in list_track_ids(root)]
    return max(lengths, default=0)


# ---------------------------------------------------------------------------
# Project lifecycle
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"


def create_project(name: str, profile_name: str, profile_attrs: dict) -> dict:
    project_id = f"{_slugify(name)}-{uuid.uuid4().hex[:8]}"
    project_dir(project_id).mkdir(parents=True, exist_ok=False)
    renders_dir(project_id).mkdir(exist_ok=True)

    mlt = ET.Element("mlt", {
        "LC_NUMERIC": "C",
        "version": MLT_VERSION,
        "root": str(project_dir(project_id)),
    })
    ET.SubElement(mlt, "profile", {k: str(v) for k, v in profile_attrs.items()})
    ET.SubElement(mlt, "tractor", {"id": "main_tractor"})
    save_tree(project_id, ET.ElementTree(mlt))

    meta = {
        "name": name,
        "profile": profile_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_meta(project_id, meta)
    return {"project_id": project_id, "name": name, "profile": profile_name}


def list_projects() -> list[dict]:
    projects = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        project_id = d.name
        if not project_exists(project_id):
            continue
        meta = load_meta(project_id)
        root = load_tree(project_id).getroot()
        track_ids = list_track_ids(root)
        num_clips = sum(
            1
            for tid in track_ids
            for c in _timeline_children(find_track(root, tid))
            if c.tag == "entry"
        )
        projects.append({
            "project_id": project_id,
            "name": meta.get("name", project_id),
            "profile": meta.get("profile"),
            "created_at": meta.get("created_at"),
            "num_tracks": len(track_ids),
            "num_clips": num_clips,
        })
    return projects


def delete_project(project_id: str) -> bool:
    if not project_exists(project_id):
        return False
    shutil.rmtree(project_dir(project_id))
    return True


def summarize(root: ET.Element) -> dict:
    fps = get_profile_fps(root)
    tractor = root.find("tractor")
    tracks_summary = []
    for idx, track_id in enumerate(list_track_ids(root)):
        playlist = find_track(root, track_id)
        timeline_children = _timeline_children(playlist)
        clips = []
        position_frames = 0
        for i, child in enumerate(timeline_children):
            if child.tag == "blank":
                position_frames += int(child.get("length"))
                continue
            producer = find_by_id(root, child.get("producer"))
            in_f, out_f = int(child.get("in")), int(child.get("out"))
            length = out_f - in_f + 1
            clips.append({
                "clip_index": i,
                "source_resource": get_property(producer, "resource"),
                "source_service": get_property(producer, "mlt_service"),
                "clip_in_seconds": frames_to_seconds(in_f, fps),
                "clip_out_seconds": frames_to_seconds(out_f, fps),
                "timeline_start_seconds": frames_to_seconds(position_frames, fps),
                "timeline_end_seconds": frames_to_seconds(position_frames + length, fps),
                "filter_ids": [f.get("id") for f in child.findall("filter")],
            })
            position_frames += length
        tracks_summary.append({
            "track_id": track_id,
            "stack_position": idx,
            "kind": get_property(playlist, KIND_PROPERTY),
            "clips": clips,
            "filter_ids": [f.get("id") for f in playlist.findall("filter")],
        })

    transitions_summary = [
        {
            "transition_id": t.get("id"),
            "service": get_property(t, "mlt_service"),
            "a_track": get_property(t, "a_track"),
            "b_track": get_property(t, "b_track"),
        }
        for t in tractor.findall("transition")
    ]

    return {
        "tracks": tracks_summary,
        "transitions": transitions_summary,
        "project_filter_ids": [f.get("id") for f in tractor.findall("filter")],
        "duration_seconds": frames_to_seconds(project_duration_frames(root), fps),
    }


# ---------------------------------------------------------------------------
# Producers
# ---------------------------------------------------------------------------

def find_or_create_producer(
    root: ET.Element,
    resource: str,
    service: str,
    out_frame: int,
    extra_props: dict[str, str] | None = None,
) -> str:
    for producer in root.findall("producer"):
        if get_property(producer, "resource") == resource and get_property(producer, "mlt_service") == service:
            if out_frame > int(producer.get("out", 0)):
                producer.set("out", str(out_frame))
            return producer.get("id")

    producer_id = new_id(root, "producer")
    producer = ET.Element("producer", {"id": producer_id, "in": "0", "out": str(out_frame)})
    set_property(producer, "resource", resource)
    set_property(producer, "mlt_service", service)
    for k, v in (extra_props or {}).items():
        set_property(producer, k, v)
    root.append(producer)
    return producer_id


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def add_track(project_id: str, kind: str, position: int | None) -> dict:
    tree = load_tree(project_id)
    root = tree.getroot()
    tractor = root.find("tractor")
    track_refs = tractor.findall("track")

    track_id = new_id(root, "track")
    playlist = ET.Element("playlist", {"id": track_id})
    if kind:
        set_property(playlist, KIND_PROPERTY, kind)
    root.append(playlist)

    track_ref = ET.Element("track", {"producer": track_id})
    if position is None or position >= len(track_refs):
        tractor.append(track_ref)
        final_position = len(track_refs)
    else:
        position = max(0, position)
        anchor_index = list(tractor).index(track_refs[position])
        tractor.insert(anchor_index, track_ref)
        final_position = position

    save_tree(project_id, tree)
    return {"track_id": track_id, "position": final_position}


def remove_track(project_id: str, track_id: str) -> None:
    tree = load_tree(project_id)
    root = tree.getroot()
    playlist = find_track(root, track_id)
    tractor = root.find("tractor")
    track_refs = tractor.findall("track")
    removed_index = next(
        (i for i, tr in enumerate(track_refs) if tr.get("producer") == track_id), None
    )
    if removed_index is None:
        raise TrackNotFoundError(f"track {track_id!r} not attached to tractor")

    for transition in list(tractor.findall("transition")):
        a_track = get_property(transition, "a_track")
        b_track = get_property(transition, "b_track")
        a_idx = int(a_track) if a_track is not None else None
        b_idx = int(b_track) if b_track is not None else None
        if a_idx == removed_index or b_idx == removed_index:
            tractor.remove(transition)
            continue
        if a_idx is not None and a_idx > removed_index:
            set_property(transition, "a_track", str(a_idx - 1))
        if b_idx is not None and b_idx > removed_index:
            set_property(transition, "b_track", str(b_idx - 1))

    tractor.remove(track_refs[removed_index])
    root.remove(playlist)
    save_tree(project_id, tree)


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

def add_clip(
    project_id: str,
    track_id: str,
    producer_service: str,
    producer_resource: str,
    clip_in_seconds: float,
    clip_out_seconds: float,
    position_seconds: float | None,
    producer_extra_props: dict[str, str] | None = None,
) -> dict:
    tree = load_tree(project_id)
    root = tree.getroot()
    fps = get_profile_fps(root)
    playlist = find_track(root, track_id)

    clip_in_frames = seconds_to_frames(clip_in_seconds, fps)
    clip_out_frames = seconds_to_frames(clip_out_seconds, fps) - 1
    if clip_out_frames < clip_in_frames:
        raise ValueError("clip_out must be greater than clip_in")

    producer_id = find_or_create_producer(
        root, producer_resource, producer_service, clip_out_frames, producer_extra_props
    )

    track_length = playlist_length_frames(playlist)
    position_frames = (
        seconds_to_frames(position_seconds, fps) if position_seconds is not None else track_length
    )
    if position_frames > track_length:
        ET.SubElement(playlist, "blank", {"length": str(position_frames - track_length)})
    elif position_frames < track_length:
        raise ValueError(
            f"position {frames_to_seconds(position_frames, fps):.3f}s overlaps existing content "
            f"(track currently extends to {frames_to_seconds(track_length, fps):.3f}s); "
            "use move_clip/trim_clip/remove_clip to rearrange instead"
        )

    entry = ET.SubElement(playlist, "entry", {
        "producer": producer_id,
        "in": str(clip_in_frames),
        "out": str(clip_out_frames),
    })
    save_tree(project_id, tree)

    clip_length = clip_out_frames - clip_in_frames + 1
    return {
        "clip_index": _timeline_children(playlist).index(entry),
        "track_id": track_id,
        "timeline_start_seconds": frames_to_seconds(position_frames, fps),
        "timeline_end_seconds": frames_to_seconds(position_frames + clip_length, fps),
    }


def _get_entry(playlist: ET.Element, clip_index: int) -> ET.Element:
    timeline_children = _timeline_children(playlist)
    if clip_index < 0 or clip_index >= len(timeline_children):
        raise ClipNotFoundError(f"clip_index {clip_index} out of range (track has {len(timeline_children)} entries)")
    entry = timeline_children[clip_index]
    if entry.tag != "entry":
        raise ClipNotFoundError(f"clip_index {clip_index} is a blank gap, not a clip")
    return entry


def trim_clip(
    project_id: str,
    track_id: str,
    clip_index: int,
    clip_in_seconds: float | None,
    clip_out_seconds: float | None,
) -> dict:
    tree = load_tree(project_id)
    root = tree.getroot()
    fps = get_profile_fps(root)
    playlist = find_track(root, track_id)
    entry = _get_entry(playlist, clip_index)

    new_in = seconds_to_frames(clip_in_seconds, fps) if clip_in_seconds is not None else int(entry.get("in"))
    new_out = seconds_to_frames(clip_out_seconds, fps) - 1 if clip_out_seconds is not None else int(entry.get("out"))
    if new_out < new_in:
        raise ValueError("clip_out must be greater than clip_in")

    entry.set("in", str(new_in))
    entry.set("out", str(new_out))
    save_tree(project_id, tree)
    return {
        "clip_in_seconds": frames_to_seconds(new_in, fps),
        "clip_out_seconds": frames_to_seconds(new_out + 1, fps),
        "duration_seconds": frames_to_seconds(new_out - new_in + 1, fps),
    }


def remove_clip(project_id: str, track_id: str, clip_index: int) -> None:
    tree = load_tree(project_id)
    root = tree.getroot()
    playlist = find_track(root, track_id)
    entry = _get_entry(playlist, clip_index)

    length = int(entry.get("out")) - int(entry.get("in")) + 1
    child_index = list(playlist).index(entry)
    playlist.remove(entry)
    playlist.insert(child_index, ET.Element("blank", {"length": str(length)}))
    save_tree(project_id, tree)


def move_clip(
    project_id: str,
    track_id: str,
    clip_index: int,
    new_position_seconds: float,
    new_track_id: str | None,
) -> dict:
    tree = load_tree(project_id)
    root = tree.getroot()
    fps = get_profile_fps(root)
    src_playlist = find_track(root, track_id)
    entry = _get_entry(src_playlist, clip_index)

    length = int(entry.get("out")) - int(entry.get("in")) + 1
    child_index = list(src_playlist).index(entry)
    src_playlist.remove(entry)
    src_playlist.insert(child_index, ET.Element("blank", {"length": str(length)}))

    dest_playlist = find_track(root, new_track_id) if new_track_id else src_playlist
    dest_length = playlist_length_frames(dest_playlist)
    new_position_frames = seconds_to_frames(new_position_seconds, fps)
    if new_position_frames > dest_length:
        ET.SubElement(dest_playlist, "blank", {"length": str(new_position_frames - dest_length)})
    elif new_position_frames < dest_length:
        raise ValueError(
            f"position {new_position_seconds:.3f}s overlaps existing content on destination track"
        )
    dest_playlist.append(entry)

    save_tree(project_id, tree)
    final_track_id = new_track_id or track_id
    return {
        "track_id": final_track_id,
        "clip_index": _timeline_children(dest_playlist).index(entry),
        "timeline_start_seconds": frames_to_seconds(new_position_frames, fps),
    }


# ---------------------------------------------------------------------------
# Transitions and filters
# ---------------------------------------------------------------------------

def add_transition(
    project_id: str,
    track_a: str,
    track_b: str,
    service: str,
    properties: dict[str, str] | None,
) -> str:
    tree = load_tree(project_id)
    root = tree.getroot()
    tractor = root.find("tractor")
    track_ids = list_track_ids(root)
    for tid in (track_a, track_b):
        if tid not in track_ids:
            raise TrackNotFoundError(f"no track {tid!r} attached to tractor")

    transition_id = new_id(root, "transition")
    transition = ET.SubElement(tractor, "transition", {"id": transition_id})
    set_property(transition, "mlt_service", service)
    set_property(transition, "a_track", str(track_ids.index(track_a)))
    set_property(transition, "b_track", str(track_ids.index(track_b)))
    for k, v in (properties or {}).items():
        set_property(transition, k, v)

    save_tree(project_id, tree)
    return transition_id


def add_filter(
    project_id: str,
    target: str,
    service: str,
    properties: dict[str, str] | None,
    clip_index: int | None,
) -> str:
    tree = load_tree(project_id)
    root = tree.getroot()

    if target == "project":
        if clip_index is not None:
            raise ValueError("clip_index is not allowed when target is 'project'")
        host = root.find("tractor")
    elif target.startswith("track:"):
        if clip_index is not None:
            raise ValueError(
                "clip_index is not allowed with a 'track:' prefix (that always targets the "
                "whole track); pass the bare track_id instead to target one clip"
            )
        host = find_track(root, target.split(":", 1)[1])
    else:
        if clip_index is None:
            raise ValueError("clip_index is required when target is a bare track_id")
        host = _get_entry(find_track(root, target), clip_index)

    filter_id = new_id(root, "filter")
    filter_el = ET.SubElement(host, "filter", {"id": filter_id})
    set_property(filter_el, "mlt_service", service)
    for k, v in (properties or {}).items():
        set_property(filter_el, k, v)

    save_tree(project_id, tree)
    return filter_id


def remove_filter(project_id: str, filter_id: str) -> None:
    tree = load_tree(project_id)
    root = tree.getroot()
    filter_el = find_by_id(root, filter_id)
    if filter_el.tag != "filter":
        raise ElementNotFoundError(f"{filter_id!r} is not a filter")
    parent = find_parent(root, filter_el)
    if parent is None:
        raise ElementNotFoundError(f"could not locate parent of filter {filter_id!r}")
    parent.remove(filter_el)
    save_tree(project_id, tree)


# ---------------------------------------------------------------------------
# Raw XML escape hatch
# ---------------------------------------------------------------------------

def set_raw_xml_property(project_id: str, element_id: str, name: str, value: str) -> None:
    tree = load_tree(project_id)
    root = tree.getroot()
    element = root if element_id == "mlt" else find_by_id(root, element_id)
    set_property(element, name, value)
    save_tree(project_id, tree)


def remove_raw_xml_property(project_id: str, element_id: str, name: str) -> bool:
    tree = load_tree(project_id)
    root = tree.getroot()
    element = root if element_id == "mlt" else find_by_id(root, element_id)
    removed = remove_property(element, name)
    save_tree(project_id, tree)
    return removed


def inject_raw_xml(project_id: str, parent_id: str, xml_fragment: str, index: int | None) -> str:
    tree = load_tree(project_id)
    root = tree.getroot()
    parent = root if parent_id == "mlt" else find_by_id(root, parent_id)

    fragment_el = ET.fromstring(xml_fragment)
    if fragment_el.get("id") is None:
        fragment_el.set("id", new_id(root, fragment_el.tag))
    if index is None:
        parent.append(fragment_el)
    else:
        parent.insert(index, fragment_el)

    save_tree(project_id, tree)
    return fragment_el.get("id")
