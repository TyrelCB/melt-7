"""Subprocess wrappers around melt-7 (render + -query introspection) and ffprobe.

melt-7's exit code is not trustworthy: confirmed empirically that it exits 0
even when a producer fails to load or an unrecognized codec name is passed
(it silently falls back). render() therefore verifies success independently
via the output file + ffprobe, and additionally cross-checks the rendered
duration against the project's expected duration -- confirmed empirically
that a missing source file can still produce a tiny-but-valid,
ffprobe-readable file (e.g. 1 frame instead of the requested 50), which a
naive "does ffprobe read it" check alone would miss.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

MELT_BIN = "melt-7"
FFPROBE_BIN = "ffprobe"

_SINGULAR = {
    "producers": "producer",
    "filters": "filter",
    "transitions": "transition",
    "consumers": "consumer",
    "profiles": "profile",
}
_LIST_ONLY_KINDS = {"formats", "video_codecs", "audio_codecs"}
_ALL_KINDS = set(_SINGULAR) | _LIST_ONLY_KINDS

_FAILURE_SUBSTRINGS = (
    "failed to load",
    "unrecognised",
    "not recognized",
    "unable to open",
)


class MeltQueryError(ValueError):
    pass


class ProbeError(RuntimeError):
    pass


def query(kind: str, service_id: str | None = None):
    """Wraps `melt-7 -query <kind>` / `-query <singular>=<service_id>`.

    Returns a list of service ids for a plain list query, or a dict of
    schema details for a single-service detail query.
    """
    if kind not in _ALL_KINDS:
        raise MeltQueryError(f"unknown kind {kind!r}; expected one of {sorted(_ALL_KINDS)}")

    if service_id is not None:
        if kind not in _SINGULAR:
            raise MeltQueryError(
                f"kind={kind!r} has no per-service detail query "
                f"(only {sorted(_SINGULAR)} support service_id)"
            )
        arg = f"{_SINGULAR[kind]}={service_id}"
    else:
        arg = kind

    proc = subprocess.run([MELT_BIN, "-query", arg], capture_output=True, text=True, timeout=15)
    data = yaml.safe_load(proc.stdout)
    if data is None:
        raise MeltQueryError(f"no metadata found for {arg!r}: {proc.stdout.strip() or proc.stderr.strip()}")
    if service_id is None and isinstance(data, dict):
        return data.get(kind, data)
    return data


def probe(file_path: str) -> dict:
    """Wraps ffprobe to inspect a media file before placing it on a timeline."""
    proc = subprocess.run(
        [FFPROBE_BIN, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", file_path],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise ProbeError(f"ffprobe failed on {file_path!r}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    fps = None
    if video and video.get("avg_frame_rate") and video["avg_frame_rate"] != "0/0":
        num, _, den = video["avg_frame_rate"].partition("/")
        try:
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            fps = None

    return {
        "duration_seconds": float(fmt["duration"]) if fmt.get("duration") else None,
        "width": video.get("width") if video else None,
        "height": video.get("height") if video else None,
        "fps": fps,
        "video_codec": video.get("codec_name") if video else None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "has_video": video is not None,
        "has_audio": audio is not None,
    }


def _scan_warnings(text: str) -> list[str]:
    warnings = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(s in lowered for s in _FAILURE_SUBSTRINGS):
            warnings.append(line.strip())
    return warnings


def render(
    project_xml_path: Path,
    output_path: Path,
    vcodec: str,
    acodec: str,
    extra_args: dict[str, str] | None,
    timeout_seconds: int,
    expected_duration_seconds: float | None,
) -> dict:
    cmd = [
        MELT_BIN, str(project_xml_path),
        "-consumer", f"avformat:{output_path}",
        f"vcodec={vcodec}",
        f"acodec={acodec}",
    ]
    for k, v in (extra_args or {}).items():
        cmd.append(f"{k}={v}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "output_path": str(output_path),
            "duration_seconds": None,
            "width": None,
            "height": None,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\nrender exceeded {timeout_seconds}s and was killed",
            "warnings": [],
            "command": cmd,
        }

    warnings = _scan_warnings(proc.stdout + "\n" + proc.stderr)

    probe_result = None
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            probe_result = probe(str(output_path))
        except ProbeError:
            probe_result = None

    success = proc.returncode == 0 and probe_result is not None and not warnings
    if success and expected_duration_seconds and probe_result.get("duration_seconds"):
        actual = probe_result["duration_seconds"]
        if actual < expected_duration_seconds * 0.9:
            success = False
            warnings.append(
                f"rendered duration ({actual:.2f}s) is far short of expected "
                f"({expected_duration_seconds:.2f}s) - a source clip likely failed to load"
            )

    return {
        "success": success,
        "output_path": str(output_path),
        "duration_seconds": probe_result.get("duration_seconds") if probe_result else None,
        "width": probe_result.get("width") if probe_result else None,
        "height": probe_result.get("height") if probe_result else None,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "warnings": warnings,
        "command": cmd,
    }
