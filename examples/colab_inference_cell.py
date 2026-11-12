# @title Viggle Remake video edit
# Copy this file's contents into one Colab cell. Everything except the four
# prompt/workflow source files is implemented inline below.

import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
from IPython.display import Video, display


VIDEO_PATH = "/content/input.mp4"  # @param {type:"string"}
PROMPT = "remove the woman"  # @param {type:"string"}
RESOLUTION = "480p"  # @param ["480p", "720p", "1080p"]
COMFY_URL = "http://127.0.0.1:8188"  # @param {type:"string"}


FPS = 24
MIN_FRAMES = 5
OFFICIAL_MIN_FRAMES = 90
OFFICIAL_MAX_FRAMES = 345
MAX_SOURCE_FRAMES = 360  # 15 seconds at 24 fps
MAX_MODEL_FRAMES = 362  # next 17n+5 length; 15.083 seconds, experimental for H3
FRAME_STEP = 17
RESOLUTION_BUCKETS = {"480p": 480, "720p": 736, "1080p": 1088}
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
MAGIC_HOUR_ROOT = "https://api.magichour.ai/v1"
MAGIC_HOUR_MODEL = "flux-2-klein"
MAGIC_HOUR_RESOLUTION = "640px"
COMFY_TIMEOUT_SECONDS = 7200
HEADINGS = (
    "subject_definitions:", "summary:", "retention_analysis:",
    "detailed_description:", "overall_soundscape:", "non_diegetic_music:",
)


def find_file(candidates, description):
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    locations = "\n".join(f"- {Path(path)}" for path in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n{locations}")


EXTRA_DIRS = (
    Path("/content/sample_data"),
    Path("/content/sample_data/viggle_e2e_prompting"),
    Path("/content/minimax_h3/viggle_e2e_prompting"),
    Path("/workspace/minimax_h3/viggle_e2e_prompting"),
    Path.cwd() / "viggle_e2e_prompting",
)
VIGGLE_DIRS = (
    Path("/content/ComfyUI/Viggle-Remake"),
    Path("/content/ComfyUI/custom_nodes/ComfyUI-Viggle-Remake"),
    Path("/content/Viggle-Remake"),
    Path("/content/ComfyUI"),
    Path("/content/minimax_h3/Viggle-Remake"),
    Path("/workspace/minimax_h3/Viggle-Remake"),
    Path.cwd() / "Viggle-Remake",
    Path.cwd() / "ComfyUI" / "Viggle-Remake",
)

IMAGE_INSTRUCTIONS_PATH = find_file(
    [directory / "GEMINI_IMAGE_EDIT_INSTRUCTIONS.md" for directory in EXTRA_DIRS],
    "GEMINI_IMAGE_EDIT_INSTRUCTIONS.md",
)
VIGGLE_INSTRUCTIONS_PATH = find_file(
    [directory / "GEMINI_VIGGLE_PROMPT_INSTRUCTIONS.md" for directory in EXTRA_DIRS],
    "GEMINI_VIGGLE_PROMPT_INSTRUCTIONS.md",
)
OFFICIAL_GUIDE_PATH = find_file(
    [directory / "PROMPTING.md" for directory in VIGGLE_DIRS],
    "the official Viggle PROMPTING.md",
)
WORKFLOW_PATH = find_file(
    [directory / "comfyui/workflows/remake_edit_i2v_v1_api.json" for directory in VIGGLE_DIRS]
    + [
        Path("/content/ComfyUI/user/default/workflows/remake_edit_i2v_v1_api.json"),
        Path("/content/ComfyUI/workflows/remake_edit_i2v_v1_api.json"),
    ],
    "the Viggle Remake edit API workflow",
)
IMAGE_INSTRUCTIONS = IMAGE_INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()
VIGGLE_INSTRUCTIONS = VIGGLE_INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()
OFFICIAL_GUIDE = OFFICIAL_GUIDE_PATH.read_text(encoding="utf-8").strip()
RUNS_ROOT = (
    Path("/content/sample_data/viggle_e2e_prompting/runs")
    if Path("/content").is_dir()
    else Path.cwd() / "viggle_e2e_prompting" / "runs"
)


def get_api_key(name):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from google.colab import userdata
        value = (userdata.get(name) or "").strip()
    except Exception:
        value = ""
    if not value:
        raise RuntimeError(f"Add {name} to Colab Secrets or set it in the environment.")
    os.environ[name] = value
    return value


def run_command(command, description):
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(f"{description}: required program is not installed: {command[0]}") from None
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"{description} failed: {detail[-3000:]}") from None
    return result.stdout


def probe_video(path):
    raw = run_command([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration:format=duration",
        "-of", "json", str(path),
    ], "Video inspection")
    info = json.loads(raw)
    streams = info.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in {path}")
    stream = streams[0]
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError(f"Could not determine video duration for {path}")
    count_text = stream.get("nb_read_frames") or "0"
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "source_fps": stream.get("avg_frame_rate"),
        "duration_seconds": duration,
        "frame_count": int(count_text) if str(count_text).isdigit() else 0,
    }


def choose_model_frame_count(source_frames):
    if source_frames < 1:
        raise ValueError("The normalized source contains no video frames.")
    if source_frames > MAX_SOURCE_FRAMES:
        raise ValueError(
            f"This single-pass workflow supports source clips up to 15 seconds "
            f"({MAX_SOURCE_FRAMES} frames at 24 fps); this input has {source_frames} frames. "
            "It will not silently discard the tail. "
            "Use a shorter selected segment or a windowed workflow."
        )
    model_frames = max(MIN_FRAMES, source_frames)
    remainder = (model_frames - 5) % FRAME_STEP
    if remainder:
        model_frames += FRAME_STEP - remainder
    if model_frames > MAX_MODEL_FRAMES:
        raise ValueError(
            f"Rounding {source_frames} source frames to the 17n+5 grid requires {model_frames} model frames, "
            f"above this workflow's {MAX_MODEL_FRAMES}-frame model-input limit. "
            "Use a shorter selected segment or a windowed workflow."
        )
    return model_frames


def make_comfy_input(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a?", "-vf", f"fps={FPS},setpts=PTS-STARTPTS",
        "-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(destination),
    ], "Preparing the ComfyUI input")


def pad_comfy_input(source, destination, source_frames, model_frames):
    padding = model_frames - source_frames
    if padding < 0:
        raise ValueError("Model frame count cannot be shorter than the selected source segment.")
    if padding == 0:
        shutil.copy2(source, destination)
        return
    vf = f"tpad=stop_mode=clone:stop={padding},trim=end_frame={model_frames},setpts=PTS-STARTPTS"
    run_command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a?", "-vf", vf, "-frames:v", str(model_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", str(destination),
    ], "Padding the ComfyUI input to the H3 frame grid")


def extract_first_frame(source, destination):
    run_command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-frames:v", "1", "-update", "1", str(destination),
    ], "Extracting source frame 0")


def gemini_generate(system_instruction, parts, response_schema, api_key):
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 9000,
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        },
    }
    response = requests.post(
        GEMINI_URL, json=payload, headers={"x-goog-api-key": api_key}, timeout=240
    )
    if not response.ok:
        raise RuntimeError(f"Gemini request failed ({response.status_code}): {response.text[:2500]}")
    body = response.json()
    candidates = body.get("candidates") or []
    response_parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
    text = "\n".join(part.get("text", "") for part in response_parts if isinstance(part, dict)).strip()
    if not text:
        raise RuntimeError(f"Gemini returned no prompt text: {body.get('promptFeedback', 'no candidate')}")
    try:
        return json.loads(text), body
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {exc}") from None


def image_content(path, label):
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [{"text": label}, {"inline_data": {"mime_type": mime_type, "data": encoded}}]


def make_image_edit_prompt(source_frame, request_text, source_meta, google_key):
    parts = image_content(source_frame, "UNTOUCHED SOURCE FRAME 0:")
    parts.append({"text": f"""User's edit request:
{request_text}

Source image dimensions: {source_meta['width']}x{source_meta['height']}.
Use only this still image and the request; do not infer later video content. Write the direct Flux 2 Klein image-edit instruction.
"""})
    schema = {
        "type": "OBJECT",
        "properties": {
            "frame_analysis": {"type": "STRING"},
            "needs_clarification": {"type": "BOOLEAN"},
            "clarification_question": {"type": "STRING"},
            "image_edit_prompt": {"type": "STRING"},
            "workflow_mode": {"type": "STRING", "enum": ["edit", "swap"]},
        },
        "required": ["frame_analysis", "needs_clarification", "clarification_question",
                     "image_edit_prompt", "workflow_mode"],
    }
    result, raw = gemini_generate(IMAGE_INSTRUCTIONS, parts, schema, google_key)
    if result.get("needs_clarification"):
        question = result.get("clarification_question", "Please clarify the edit.")
        raise RuntimeError(f"Gemini needs clarification before image editing: {question}")
    prompt = str(result.get("image_edit_prompt", "")).strip()
    if not prompt:
        raise RuntimeError("Gemini returned an empty Flux image-edit prompt.")
    if result.get("workflow_mode") != "edit":
        raise RuntimeError(
            "Gemini classified this as a person/character identity swap, but this cell loads "
            "the edit workflow only. Use a swap workflow for this request."
        )
    return prompt, result, raw


def edit_first_frame_with_magic_hour(frame_path, prompt, output_dir, magic_key):
    extension = frame_path.suffix.lower().lstrip(".") or "png"
    headers = {"Authorization": f"Bearer {magic_key}"}
    response = requests.post(
        f"{MAGIC_HOUR_ROOT}/files/upload-urls",
        json={"items": [{"type": "image", "extension": extension}]},
        headers=headers, timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"Magic Hour upload URL request failed ({response.status_code}): {response.text[:2000]}")
    item = (response.json().get("items") or [None])[0]
    if not item or not item.get("upload_url") or not item.get("file_path"):
        raise RuntimeError("Magic Hour returned an incomplete upload URL record.")
    mime_type = mimetypes.guess_type(frame_path.name)[0] or "image/png"
    upload_response = requests.put(item["upload_url"], data=frame_path.read_bytes(),
                                   headers={"Content-Type": mime_type}, timeout=240)
    if not upload_response.ok:
        raise RuntimeError(f"Magic Hour image upload failed ({upload_response.status_code}): {upload_response.text[:1500]}")
    project_response = requests.post(
        f"{MAGIC_HOUR_ROOT}/ai-image-editor",
        json={
            "name": f"Viggle frame edit {output_dir.name}",
            "image_count": 1,
            "model": MAGIC_HOUR_MODEL,
            "resolution": MAGIC_HOUR_RESOLUTION,
            "aspect_ratio": "auto",
            "style": {"prompt": prompt},
            "assets": {"image_file_paths": [item["file_path"]]},
        },
        headers=headers, timeout=60,
    )
    if not project_response.ok:
        raise RuntimeError(f"Magic Hour edit request failed ({project_response.status_code}): {project_response.text[:2000]}")
    project = project_response.json()
    project_id = project.get("id")
    if not project_id:
        raise RuntimeError(f"Magic Hour response omitted project id: {project}")

    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        time.sleep(3)
        status_response = requests.get(
            f"{MAGIC_HOUR_ROOT}/image-projects/{project_id}", headers=headers, timeout=60
        )
        if not status_response.ok:
            raise RuntimeError(f"Magic Hour status request failed ({status_response.status_code}): {status_response.text[:1500]}")
        details = status_response.json()
        status = str(details.get("status", "")).lower()
        if status in {"failed", "error", "canceled", "cancelled"}:
            raise RuntimeError(f"Magic Hour image edit failed: {details}")
        if status in {"complete", "completed"}:
            downloads = details.get("downloads") or []
            image_url = downloads[0].get("url") if downloads else None
            if not image_url:
                raise RuntimeError("Magic Hour completed without an image URL.")
            image_response = requests.get(image_url, timeout=240)
            if not image_response.ok:
                raise RuntimeError(f"Could not download edited frame ({image_response.status_code}).")
            mime = image_response.headers.get("Content-Type", "image/png").split(";", 1)[0].lower()
            suffix = ".jpg" if mime in {"image/jpeg", "image/jpg"} else ".png"
            edited_path = output_dir / f"edited_first_frame{suffix}"
            edited_path.write_bytes(image_response.content)
            return edited_path, {
                "project_id": project_id,
                "status": status,
                "model": MAGIC_HOUR_MODEL,
                "resolution": MAGIC_HOUR_RESOLUTION,
                "aspect_ratio": "auto",
                "credits_charged": project.get("credits_charged"),
                "output_file": str(edited_path),
            }
    raise TimeoutError(f"Magic Hour project {project_id} did not finish within 300 seconds.")


def format_final_prompt(prompt):
    prompt = str(prompt or "").strip()
    for heading in HEADINGS:
        prompt = re.sub(rf"\s*{re.escape(heading)}\s*", "\n" + heading + "\n", prompt)
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()


def validate_final_prompt(prompt, duration):
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    found = [line for line in lines if line in HEADINGS]
    if found != list(HEADINGS) or any(lines.count(h) != 1 for h in HEADINGS):
        raise ValueError("Gemini's final prompt must contain the six official headings once, in order.")
    if "<Picture 1>" not in prompt:
        raise ValueError("Gemini's final prompt must define the edited frame as <Picture 1>.")
    if "```" in prompt or any(ord(char) > 127 for char in prompt):
        raise ValueError("Gemini's final prompt contains markdown fences or non-ASCII characters.")
    for match in re.finditer(r"\b(\d{2}):(\d{2})\.(\d{3})\b", prompt):
        timestamp = int(match[1]) * 60 + int(match[2]) + int(match[3]) / 1000
        if timestamp > duration + 0.001:
            raise ValueError(f"Gemini timestamp {match[0]} exceeds the render duration {duration:.3f}s.")


def make_final_viggle_prompt(source_frame, edited_frame, request_text, metadata, google_key):
    parts = image_content(source_frame, "UNTOUCHED SOURCE FRAME 0:")
    parts.extend(image_content(edited_frame, "ACTUAL EDITED TARGET FRAME (define as <Picture 1>):"))
    parts.append({"text": f"""OFFICIAL VIGGLE REMAKE PROMPTING GUIDE (follow as source of truth):
{OFFICIAL_GUIDE}

User's simple edit request:
{request_text}

Source/render metadata: {json.dumps(metadata, ensure_ascii=True)}

Assume one continuous shot with no cuts. The source video itself is not attached. Use the two stills for visible details, metadata for duration, and the source video as <Video 1>. Do not claim to have analyzed motion or heard audio. The edited target still is <Picture 1>, pinned at source frame 0. Compare it with the request; if they conflict materially, set needs_clarification and ask one concise question.
"""})
    schema = {
        "type": "OBJECT",
        "properties": {
            "frame_analysis": {"type": "STRING"},
            "needs_clarification": {"type": "BOOLEAN"},
            "clarification_question": {"type": "STRING"},
            "prompt": {"type": "STRING"},
        },
        "required": ["frame_analysis", "needs_clarification", "clarification_question", "prompt"],
    }
    result, raw = gemini_generate(VIGGLE_INSTRUCTIONS, parts, schema, google_key)
    if result.get("needs_clarification"):
        question = result.get("clarification_question", "The edited reference conflicts with the request.")
        raise RuntimeError(f"Gemini found a reference/request conflict: {question}")
    prompt = format_final_prompt(result.get("prompt", ""))
    validate_final_prompt(prompt, metadata["duration_seconds"])
    return prompt, result, raw


def comfy_request(base_url, route, payload=None, timeout=60):
    url = base_url.rstrip("/") + route
    try:
        response = requests.post(url, json=payload, timeout=timeout) if payload is not None else requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"ComfyUI request failed for {route}: {exc}") from None
    if not response.ok:
        raise RuntimeError(f"ComfyUI {route} failed ({response.status_code}): {response.text[:2500]}")
    return response.json()


def upload_to_comfy(base_url, path, remote_name):
    with path.open("rb") as handle:
        response = requests.post(
            base_url.rstrip("/") + "/upload/image",
            data={"type": "input", "overwrite": "true"},
            files={"image": (remote_name, handle, mimetypes.guess_type(path.name)[0] or "application/octet-stream")},
            timeout=300,
        )
    if not response.ok:
        raise RuntimeError(f"ComfyUI upload failed ({response.status_code}): {response.text[:2500]}")
    item = response.json()
    name = item.get("name")
    if not name:
        raise RuntimeError(f"ComfyUI upload response omitted file name: {item}")
    return f"{item.get('subfolder')}/{name}".lstrip("/") if item.get("subfolder") else name


def one_node(workflow, class_type):
    matches = [(node_id, node) for node_id, node in workflow.items()
               if isinstance(node, dict) and node.get("class_type") == class_type]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {class_type} node; found {len(matches)}.")
    return matches[0]


def build_edit_workflow(source_name, reference_name, prompt, frames, short_edge, seed, prefix):
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    _, source = one_node(workflow, "LoadVideo")
    _, conditioning = one_node(workflow, "RemakeConditioning")
    conditioning["class_type"] = conditioning_node_class(frames, short_edge)
    _, noise = one_node(workflow, "RemakeNoise")
    _, save = one_node(workflow, "SaveVideo")
    source["inputs"]["file"] = source_name
    conditioning_inputs = conditioning["inputs"]
    conditioning_inputs.update({"prompt": prompt, "pins": "0", "frames": frames, "short_edge": short_edge})
    conditioning_inputs.pop("ref_2", None)
    ref_link = conditioning_inputs.get("ref_1")
    if not isinstance(ref_link, list) or ref_link[0] not in workflow:
        raise ValueError("The edit workflow's ref_1 must connect to a LoadImage node.")
    ref_node = workflow[ref_link[0]]
    if ref_node.get("class_type") != "LoadImage":
        raise ValueError("The edit workflow's ref_1 is not connected to LoadImage.")
    ref_node["inputs"]["image"] = reference_name
    for node_id, node in list(workflow.items()):
        if node.get("class_type") == "LoadImage" and node_id != ref_link[0]:
            workflow.pop(node_id)
    noise["inputs"]["seed"] = seed
    save["inputs"]["filename_prefix"] = prefix
    return workflow


def conditioning_node_class(model_frames, short_edge):
    return "RemakeConditioningExtended" if (
        model_frames < OFFICIAL_MIN_FRAMES
        or model_frames > OFFICIAL_MAX_FRAMES
        or short_edge > 736
    ) else "RemakeConditioning"


def get_video_output(job, save_node_id):
    matches = []
    save_output = (job.get("outputs") or {}).get(save_node_id)
    if not isinstance(save_output, dict):
        raise RuntimeError(f"ComfyUI history has no output for the workflow's SaveVideo node {save_node_id}.")
    for node in (save_output,):
        for value in node.values():
            if isinstance(value, list):
                matches.extend(item for item in value if isinstance(item, dict)
                               and item.get("filename", "").lower().endswith((".mp4", ".mov", ".webm")))
    if not matches:
        raise RuntimeError("ComfyUI completed without returning a video output.")
    return matches[-1]


def download_comfy_output(base_url, item, destination):
    response = requests.get(
        base_url.rstrip("/") + "/view",
        params={"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "output")},
        timeout=300,
    )
    if not response.ok:
        raise RuntimeError(f"ComfyUI output download failed ({response.status_code}).")
    destination.write_bytes(response.content)


def render_viggle(video_path, request_text, resolution, comfy_url):
    source_meta = probe_video(video_path)
    short_edge = RESOLUTION_BUCKETS[resolution]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    comfy_request(comfy_url, "/system_stats")
    source_frame = run_dir / "source_frame_000000.png"
    extract_first_frame(video_path, source_frame)
    normalized_input = run_dir / "comfy_input_source_24fps.mp4"
    make_comfy_input(video_path, normalized_input)
    normalized_meta = probe_video(normalized_input)
    source_frames = normalized_meta["frame_count"]
    frames = choose_model_frame_count(source_frames)
    source_duration = source_frames / FPS
    model_duration = frames / FPS
    comfy_input = run_dir / "comfy_input_model_24fps.mp4"
    pad_comfy_input(normalized_input, comfy_input, source_frames, frames)
    comfy_input_meta = probe_video(comfy_input)
    if comfy_input_meta["frame_count"] != frames:
        raise ValueError(f"Prepared ComfyUI input has {comfy_input_meta['frame_count']} frames; expected {frames}.")

    google_key = get_api_key("GOOGLE_API_KEY")
    magic_key = get_api_key("MAGICHOUR_API_KEY")
    print("Creating the edited frame and six-section prompt...")
    image_prompt, image_audit, _ = make_image_edit_prompt(
        source_frame, request_text, source_meta, google_key
    )
    (run_dir / "image_edit_prompt.txt").write_text(image_prompt + "\n", encoding="utf-8")
    (run_dir / "gemini_image_audit.json").write_text(json.dumps(image_audit, indent=2) + "\n", encoding="utf-8")

    edited_frame, magic_record = edit_first_frame_with_magic_hour(
        source_frame, image_prompt, run_dir, magic_key
    )
    render_meta = {
        "width": source_meta["width"], "height": source_meta["height"],
        "fps": FPS, "duration_seconds": source_duration, "frame_count_estimate": source_frames,
    }
    final_prompt, prompt_audit, _ = make_final_viggle_prompt(
        source_frame, edited_frame, request_text, render_meta, google_key
    )
    (run_dir / "viggle_prompt.txt").write_text(final_prompt + "\n", encoding="utf-8")
    (run_dir / "gemini_viggle_audit.json").write_text(json.dumps(prompt_audit, indent=2) + "\n", encoding="utf-8")

    source_name = upload_to_comfy(comfy_url, comfy_input, f"viggle_{run_id}_source.mp4")
    reference_name = upload_to_comfy(comfy_url, edited_frame, f"viggle_{run_id}_pin0{edited_frame.suffix}")
    workflow = build_edit_workflow(
        source_name, reference_name, final_prompt, frames, short_edge, 0,
        f"viggle_runs/{run_id}/render",
    )
    workflow_path = run_dir / "workflow_api.json"
    workflow_path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_node_id, _ = one_node(workflow, "SaveVideo")
    conditioning_class = conditioning_node_class(frames, short_edge)
    if conditioning_class == "RemakeConditioningExtended":
        available_nodes = comfy_request(comfy_url, "/object_info/RemakeConditioningExtended")
        if "RemakeConditioningExtended" not in available_nodes:
            raise RuntimeError(
                "This render needs RemakeConditioningExtended. Install the Viggle Remake Extended node on the ComfyUI server."
            )
    manifest = {
        "source_video": str(video_path), "source_metadata": source_meta,
        "normalized_source": str(normalized_input), "prepared_comfy_input": str(comfy_input),
        "source_frames": source_frames, "source_duration_seconds": source_duration,
        "render_duration_seconds": source_duration, "model_frames": frames,
        "model_duration_seconds": model_duration, "fps": FPS, "unused_tail_frames": 0,
        "temporal_padding_frames": frames - source_frames, "resolution": resolution, "short_edge": short_edge,
        "workflow": str(WORKFLOW_PATH), "pin_frame": 0, "sampler_seed": 0,
        "prompt": request_text, "image_edit_prompt": image_prompt,
        "edited_frame": str(edited_frame), "magic_hour": magic_record,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Submitting {frames} model frames at {resolution}; saving exactly {source_frames} frames ({source_duration:.3f}s) with a frame-0 pin...")
    queued = comfy_request(comfy_url, "/prompt", {"prompt": workflow, "client_id": str(uuid.uuid4())}, timeout=120)
    if queued.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected the workflow: {queued['node_errors']}")
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt id: {queued}")
    manifest["comfy_prompt_id"] = prompt_id
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    deadline = time.monotonic() + COMFY_TIMEOUT_SECONDS
    last_notice = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(5)
        history = comfy_request(comfy_url, "/history/" + prompt_id, timeout=60)
        job = history.get(prompt_id)
        if not job:
            continue
        status = job.get("status") or {}
        if status.get("status_str") == "error":
            (run_dir / "comfy_error.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"ComfyUI inference failed: {status.get('messages', status)}")
        if status.get("completed") and job.get("outputs"):
            break
        if time.monotonic() - last_notice >= 60:
            print("ComfyUI is still rendering...")
            last_notice = time.monotonic()
    else:
        raise TimeoutError(f"ComfyUI job {prompt_id} did not finish within {COMFY_TIMEOUT_SECONDS} seconds.")

    item = get_video_output(job, save_node_id)
    raw_output = run_dir / ("comfy_output_raw" + Path(item["filename"]).suffix.lower())
    download_comfy_output(comfy_url, item, raw_output)
    output_path = run_dir / "output.mp4"
    run_command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_output),
        "-map", "0:v:0", "-map", "0:a?", "-t", f"{source_duration:.6f}", "-frames:v", str(source_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output_path),
    ], "Saving the rendered video")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("ComfyUI finished without a usable MP4 file.")
    output_meta = probe_video(output_path)
    if output_meta["frame_count"] != source_frames:
        raise RuntimeError(f"Final output has {output_meta['frame_count']} frames; expected exactly {source_frames}.")
    if abs(output_meta["duration_seconds"] - source_duration) > (1 / FPS):
        raise RuntimeError(
            f"Final output duration is {output_meta['duration_seconds']:.3f}s; expected {source_duration:.3f}s."
        )
    manifest.update({"output": str(output_path), "comfy_output": item,
                     "output_metadata": output_meta,
                     "completed_at": datetime.now().isoformat(timespec="seconds")})
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


if not PROMPT.strip():
    raise ValueError("Enter an edit prompt.")
if RESOLUTION not in RESOLUTION_BUCKETS:
    raise ValueError("Choose 480p, 720p, or 1080p.")
if not COMFY_URL.strip():
    raise ValueError("Enter the ComfyUI server URL.")

source_video = Path(VIDEO_PATH).expanduser()
if not source_video.is_absolute():
    source_video = Path.cwd() / source_video
if not source_video.is_file():
    raise FileNotFoundError(f"Video file not found: {source_video}")

print(f"Viggle Remake · {RESOLUTION} · {source_video.name}")
output_video = render_viggle(source_video.resolve(), PROMPT.strip(), RESOLUTION, COMFY_URL.strip().rstrip("/"))
print(f"Saved: {output_video}")
display(Video(str(output_video), embed=True, width=640))
