"""Hosted vision-language "oracle" for sparse keyframes.

A multimodal model looks at one frame every few seconds and returns JSON: every on-field person
with a box, team (jersey colour) and role, the ball, and named pitch landmarks. Local models do
the dense per-frame work in between; the oracle supplies what they are bad at (who is a player
vs a spectator, which team, who is a referee or goalkeeper) and a calibration seed when the
field detector fails.

Measured on 4 frames (day, wide far-side, night, multi-sport; 2026-10-08): the cheap tier found
essentially every on-field player, both goalkeepers and the referees, ignored spectators and the
bench, and its boxes matched YOLO's within a few pixels at the feet. The landmark tier's
coordinates produced a homography within ~0.9 m (median) of the field detector's on a daytime
frame. JSON coordinates beat redrawn images: same scene understanding, ~25x cheaper, tighter.

Two more request kinds serve FastPipeline: `submit_markings` (every visible soccer marking with a
confidence, for the VLM field-marking calibration in `calibration.field_markings`; needs the
stronger tier) and `submit_ball` (is the ball in this crop, for the ball tracker's sparse anchors).
`decide` asks the OpenAI Decisions API (POST /v1/decisions, gpt-6-luna; public beta) yes/no
questions about images and gets probabilities back, never coordinates or text: ~$0.0002 and
~0.8 s per 1920-px image, extra questions on the same image nearly free. FastPipeline uses it
for per-track "is this person taking part in the match" answers (`onfield.ask_participants`).

Model-agnostic by design: pipelines depend on `VisionOracle` (submit_people/submit_landmarks/
submit_markings/submit_ball returning futures of `OracleResult`). `OpenAIVisionOracle` is one implementation; the API key
comes from the environment and is never logged. Every call is costed from the reported usage
against a hard per-run budget; once it is reached, further calls are skipped (returning None).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from soccer_analyzer import ball_track as _bt
from soccer_analyzer.calibration import field_markings as _fm
from soccer_analyzer.calibration import joint_calib as _jc
from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH

# USD per 1M tokens (input, output), developers.openai.com/api/docs/pricing, 2026-10-08.
OPENAI_PRICES = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6-sol": (2.0, 10.0),
    "gpt-6-astra": (10.0, 50.0),
    "gpt-5.4-mini": (0.75, 4.50),
}
# Tokens reserved per image call against the budget (input, output); see `_reserve`.
RESERVE_TOKENS = (3000, 1900)
# Joint field calibration asks 2-8 markings keyframes per camera setup (+2 yard-number reads on
# football-lined fields), each reserved at $0.025 with gpt-6-sol: 2 keyframes need >= $0.05 of
# budget left after the people keyframes, a football-lined field with 4 keyframes >= $0.15.
MIN_USD_JOINT = 0.05
MIN_USD_JOINT_FOOTBALL = 0.15
DECISIONS_PRICE_IN = 0.10  # USD per 1M input tokens; /v1/decisions bills input only
_UNKNOWN_PRICE = (10.0, 50.0)  # budget conservatively for unlisted models

# Named landmarks in the standard 105 x 68 m corner-origin frame, as seen by a camera on the
# near touchline: x runs left -> right along the length, y from the far touchline (0) to the
# near touchline (68). "left"/"right" are the image sides. The centre-circle extremes are image
# extremes approximated by the circle's x-extremes (small error under perspective).
_END_POINTS = {
    "corner_far": (0.0, 0.0, "corner of the field where the goal line meets the FAR touchline"),
    "corner_near": (0.0, 68.0, "corner of the field where the goal line meets the NEAR touchline"),
    "box_goal_far": (0.0, 13.84, "penalty area: where its far side line meets the goal line"),
    "box_goal_near": (0.0, 54.16, "penalty area: where its near side line meets the goal line"),
    "box_front_far": (16.5, 13.84, "penalty area: front corner on the far side"),
    "box_front_near": (16.5, 54.16, "penalty area: front corner on the near side"),
    "six_goal_far": (0.0, 24.84, "goal area: where its far side line meets the goal line"),
    "six_goal_near": (0.0, 43.16, "goal area: where its near side line meets the goal line"),
    "six_front_far": (5.5, 24.84, "goal area: front corner on the far side"),
    "six_front_near": (5.5, 43.16, "goal area: front corner on the near side"),
    "penalty_spot": (11.0, 34.0, "penalty spot"),
    "arc_far": (16.5, 26.68, "where the penalty arc meets the penalty-area front line (far side)"),
    "arc_near": (
        16.5,
        41.31,
        "where the penalty arc meets the penalty-area front line (near side)",
    ),
}
LANDMARKS: Dict[str, Tuple[float, float, str]] = {
    "halfway_far": (52.5, 0.0, "where the halfway line meets the FAR touchline"),
    "halfway_near": (52.5, 68.0, "where the halfway line meets the NEAR touchline"),
    "centre_spot": (52.5, 34.0, "the centre spot"),
    "circle_far": (52.5, 24.85, "where the halfway line crosses the far side of the centre circle"),
    "circle_near": (
        52.5,
        43.15,
        "where the halfway line crosses the near side of the centre circle",
    ),
    "circle_left": (43.35, 34.0, "the left-most point of the centre circle in the image"),
    "circle_right": (61.65, 34.0, "the right-most point of the centre circle in the image"),
}
for _side, _mirror in (("left", False), ("right", True)):
    for _key, (_x, _y, _desc) in _END_POINTS.items():
        LANDMARKS[f"{_side}_{_key}"] = (105.0 - _x if _mirror else _x, _y, f"{_side} end: {_desc}")

PEOPLE_PROMPT = (
    "This soccer image is {W}x{H} pixels. Find EVERY person on the field: all players (including "
    "tiny distant ones and goalkeepers) and referees/assistant referees. Ignore spectators, "
    'coaches and substitutes off the field. Return JSON: {{"people": [{{"role": '
    '"player|goalkeeper|referee", "team": "<jersey colour>" or null, "box": [x1, y1, x2, '
    'y2]}}], "ball": [x, y] or null}} in absolute pixel coordinates of this {W}x{H} image.'
)


def _landmark_prompt() -> str:
    names = "\n".join(f"- {k}: {d}" for k, (_, _, d) in LANDMARKS.items())
    return (
        "This soccer image is {W}x{H} pixels, filmed from beside one touchline (the NEAR "
        "touchline is the one closest to the camera; left/right mean image sides). Locate the "
        "painted pitch landmarks below that are clearly visible and return JSON "
        '{{"landmarks": [{{"id": "<id>", "xy": [x, y]}}], "ball": [x, y] or null}} in absolute '
        "pixel coordinates of this image. Use only these ids, omit landmarks you cannot see, "
        "and be as precise as possible:\n" + names.replace("{", "{{").replace("}", "}}")
    )


LANDMARK_PROMPT = _landmark_prompt()


def _format_safe(prompt: str) -> str:
    """Escape JSON braces so `str.format(W=..., H=...)` only fills {W} and {H}."""
    s = prompt.replace("{", "{{").replace("}", "}}")
    return s.replace("{{W}}", "{W}").replace("{{H}}", "{H}")


MARKINGS_PROMPT = _format_safe(_fm.PROMPT)
NUMBERS_PROMPT = _format_safe(_jc.PROMPT_NUMBERS)
BALL_PROMPT = _format_safe(_bt.BALL_PROMPT)


@dataclass(frozen=True)
class OraclePerson:
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in ORIGINAL frame pixels
    role: str  # "player" | "goalkeeper" | "referee"
    team: Optional[str]  # free-text jersey colour, lower-case; None for officials


@dataclass
class OracleResult:
    """One keyframe answer, in original-frame pixel coordinates."""

    frame_idx: int
    kind: str  # "people" | "landmarks" | "markings" | "ball..." (crop question)
    people: List[OraclePerson] = field(default_factory=list)
    ball: Optional[Tuple[float, float]] = None  # for "ball" questions: in the sent image's pixels
    landmarks: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    markings: List[Tuple[str, float, float, float]] = field(default_factory=list)  # name,x,y,conf
    line_colour: Optional[str] = None  # soccer line colour named with the markings
    # American-football yard numbers read (kind "numbers"): dict(value, x, y, toward, side,
    # confidence) in original pixels
    numbers: List[dict] = field(default_factory=list)
    latency_s: float = 0.0
    cost_usd: float = 0.0
    model: str = ""


class VisionOracle(Protocol):
    def submit_people(self, frame_idx: int, frame: np.ndarray) -> Optional[Future]: ...

    def submit_landmarks(self, frame_idx: int, frame: np.ndarray) -> Optional[Future]: ...

    def submit_markings(
        self, frame_idx: int, frame: np.ndarray, alt: bool = False
    ) -> Optional[Future]: ...

    def submit_ball(self, frame_idx: int, image: np.ndarray, tag: str = "") -> Optional[Future]: ...

    @property
    def spent_usd(self) -> float: ...

    def close(self) -> None: ...


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _point(v, sx: float, sy: float, w: int, h: int) -> Optional[Tuple[float, float]]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    x, y = _num(v[0]), _num(v[1])
    if x is None or y is None or not (0 <= x <= w / sx and 0 <= y <= h / sy):
        return None
    return x * sx, y * sy


def parse_people(payload: dict, sx: float, sy: float, w: int, h: int):
    """Validated people + ball from a JSON reply. sx, sy scale reply pixels to original pixels;
    w, h are the original frame size. Malformed entries are dropped, boxes clipped to the frame."""
    people: List[OraclePerson] = []
    for p in payload.get("people") or []:
        if not isinstance(p, dict):
            continue
        b = p.get("box")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        v = [_num(x) for x in b]
        if any(x is None for x in v):
            continue
        x1, x2 = sorted((v[0] * sx, v[2] * sx))
        y1, y2 = sorted((v[1] * sy, v[3] * sy))
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        if x2 - x1 < 2 or y2 - y1 < 4:
            continue
        role = str(p.get("role") or "player").lower()
        role = role if role in ("player", "goalkeeper", "referee") else "player"
        team = p.get("team")
        team = str(team).strip().lower() if team and role != "referee" else None
        people.append(OraclePerson((x1, y1, x2, y2), role, team))
    return people, _point(payload.get("ball"), sx, sy, w, h)


def parse_landmarks(payload: dict, sx: float, sy: float, w: int, h: int):
    """Validated {landmark_id: (x, y)} (original pixels) + ball. Unknown ids are dropped."""
    out: Dict[str, Tuple[float, float]] = {}
    for item in payload.get("landmarks") or []:
        if not isinstance(item, dict) or item.get("id") not in LANDMARKS:
            continue
        p = _point(item.get("xy"), sx, sy, w, h)
        if p is not None:
            out[item["id"]] = p
    return out, _point(payload.get("ball"), sx, sy, w, h)


def landmark_homography(
    landmarks: Dict[str, Tuple[float, float]],
    pitch: PitchModel = STANDARD_PITCH,
    ransac_px: float = 12.0,
    min_points: int = 4,
) -> Optional[np.ndarray]:
    """Image(px) -> pitch(m) homography from named oracle landmarks, or None.

    Needs >= min_points correspondences in general position (not collinear) in both image and
    pitch; residuals are measured in image pixels. Landmark coordinates are defined on the
    standard pitch and rescaled to `pitch` like the field detector's keypoints."""
    from soccer_analyzer.calibration.keypoint_calib import _general_position

    if len(landmarks) < min_points:
        return None
    sx, sy = pitch.length / 105.0, pitch.width / 68.0
    img = np.array([landmarks[k] for k in landmarks], np.float32)
    world = np.array([[LANDMARKS[k][0] * sx, LANDMARKS[k][1] * sy] for k in landmarks], np.float32)
    if not _general_position(img) or not _general_position(world):
        return None
    inverse, mask = cv2.findHomography(world, img, cv2.RANSAC, ransac_px)
    if inverse is None or mask is None or int(mask.sum()) < min_points:
        return None
    try:
        H = np.linalg.inv(inverse)
    except np.linalg.LinAlgError:
        return None
    return H if np.all(np.isfinite(H)) else None


class OpenAIVisionOracle:
    """VisionOracle backed by the OpenAI Responses API (JSON mode).

    Calls run on a thread pool so their 7-20 s latency overlaps local processing. `max_usd` is a
    hard cap per instance: a call is only issued if the running cost plus a reservation for it
    stays under the cap; the reservation is replaced by the cost reported in the response."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        people_model: str = "gpt-6-luna",
        landmark_model: str = "gpt-6-sol",
        max_usd: float = 0.50,
        markings_model: str = "gpt-6-sol",
        markings_alt_model: str = "gpt-6-luna",
        ball_model: str = "gpt-6-luna",
        max_workers: int = 6,
        image_width: int = 1920,
        reasoning_effort: str = "low",
        client=None,
        decide_fn=None,
        decisions_model: str = "gpt-6-luna",
    ):
        if max_usd <= 0 or max_workers <= 0 or image_width < 256:
            raise ValueError("max_usd and max_workers must be positive; image_width >= 256")
        self._api_key = api_key
        if client is None:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set; the vision oracle is disabled.")
            from openai import OpenAI

            client = OpenAI(api_key=key)
        self._client = client
        # decide_fn(images, questions, text=, max_width=) -> answers dict: an injected Decisions
        # transport (tests, budgeted lab runners); default = HTTPS with OPENAI_API_KEY.
        self._decide_fn = decide_fn
        self.decisions_model = decisions_model
        self.people_model = people_model
        self.landmark_model = landmark_model
        self.markings_model = markings_model
        self.markings_alt_model = markings_alt_model
        self.ball_model = ball_model
        self.max_usd = float(max_usd)
        self.image_width = int(image_width)
        self.reasoning_effort = reasoning_effort
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="oracle")
        self._lock = threading.Lock()
        self._spent = 0.0
        self._reserved = 0.0
        self.calls: List[dict] = []  # per-call log: model, kind, frame, latency, cost (no data)

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent

    def _reserve(self, model: str) -> Optional[float]:
        pin, pout = OPENAI_PRICES.get(model, _UNKNOWN_PRICE)
        # Realistic upper bound for one 1920-px image call: ~3000 input tokens and <= 1900
        # output tokens. Sol: $0.025 (measured $0.012-0.021 per markings / yard-number call;
        # the old 4000+4000 bound reserved $0.048, which let a $0.10 budget ask only 2
        # keyframes). The actual usage replaces the reservation when the call returns.
        reserve = (RESERVE_TOKENS[0] * pin + RESERVE_TOKENS[1] * pout) / 1e6
        with self._lock:
            if self._spent + self._reserved + reserve > self.max_usd:
                return None
            self._reserved += reserve
        return reserve

    def submit_people(self, frame_idx: int, frame: np.ndarray) -> Optional[Future]:
        return self._submit(frame_idx, frame, "people", self.people_model, PEOPLE_PROMPT)

    def submit_landmarks(self, frame_idx: int, frame: np.ndarray) -> Optional[Future]:
        return self._submit(frame_idx, frame, "landmarks", self.landmark_model, LANDMARK_PROMPT)

    def submit_markings(
        self, frame_idx: int, frame: np.ndarray, alt: bool = False
    ) -> Optional[Future]:
        """Every visible soccer marking point with a confidence (field_markings.PROMPT), from the
        markings model, or with `alt` from the second (cheap) model. On its own the cheap tier's
        markings were too unreliable to fit a camera (lab 2026-10-12); as a second opinion its
        list wins when it names more markings (the rule the lab's window results used)."""
        if alt:
            return self._submit(
                frame_idx, frame, "markings_alt", self.markings_alt_model, MARKINGS_PROMPT
            )
        return self._submit(frame_idx, frame, "markings", self.markings_model, MARKINGS_PROMPT)

    def submit_numbers(self, frame_idx: int, frame: np.ndarray) -> Optional[Future]:
        """The American-football yard numbers painted on a multi-sport field, from the markings
        model (`joint_calib.PROMPT_NUMBERS`): they fix the football grid's 5-yard period."""
        return self._submit(frame_idx, frame, "numbers", self.markings_model, NUMBERS_PROMPT)

    def submit_ball(self, frame_idx: int, image: np.ndarray, tag: str = "") -> Optional[Future]:
        """Is the ball in `image` (a crop of frame `frame_idx`)? `tag` names the crop (cache key).
        The answer's `ball` is in `image` pixels, None when the model saw no ball."""
        return self._submit(frame_idx, image, "ball" + tag, self.ball_model, BALL_PROMPT)

    def model_for(self, kind: str) -> str:
        if kind == "people":
            return self.people_model
        if kind in ("markings", "numbers"):
            return self.markings_model
        if kind == "markings_alt":
            return self.markings_alt_model
        if kind.startswith("ball"):
            return self.ball_model
        return self.landmark_model

    def _submit(self, frame_idx, frame, kind, model, prompt) -> Optional[Future]:
        reserve = self._reserve(model)
        if reserve is None:
            return None
        h, w = frame.shape[:2]
        scale = min(1.0, self.image_width / float(w))
        size = (int(round(w * scale)), int(round(h * scale)))
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            with self._lock:
                self._reserved -= reserve
            return None
        sh, sw = small.shape[:2]
        text = prompt.format(W=sw, H=sh)
        b64 = base64.b64encode(jpg.tobytes()).decode()
        return self._pool.submit(
            self._call, frame_idx, kind, model, text, b64, (w / sw, h / sh, w, h), reserve
        )

    def _call(self, frame_idx, kind, model, text, b64, geom, reserve) -> Optional[OracleResult]:
        sx, sy, w, h = geom
        t0 = time.time()
        cost = 0.0
        try:
            r = self._client.responses.create(
                model=model,
                reasoning={"effort": self.reasoning_effort},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": text},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high",
                            },
                        ],
                    }
                ],
                text={"format": {"type": "json_object"}},
            )
            pin, pout = OPENAI_PRICES.get(model, _UNKNOWN_PRICE)
            usage = getattr(r, "usage", None)
            if usage is not None:
                cost = (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6
            else:
                cost = reserve
            payload = json.loads(r.output_text or "{}")
        except Exception as exc:  # network/API/JSON failure -> no answer for this keyframe
            payload, err = None, type(exc).__name__
        else:
            err = None
        finally:
            with self._lock:
                self._reserved -= reserve
                self._spent += cost
        latency = time.time() - t0
        self.calls.append(
            dict(frame=frame_idx, kind=kind, model=model, secs=round(latency, 1), usd=cost, err=err)
        )
        if not isinstance(payload, dict):
            return None
        res = OracleResult(frame_idx, kind, latency_s=latency, cost_usd=cost, model=model)
        if kind == "people":
            res.people, res.ball = parse_people(payload, sx, sy, w, h)
        elif kind.startswith("markings"):
            res.markings = _fm.parse_markings(payload, scale=(sx, sy))
            res.line_colour = str(payload.get("soccer_line_colour") or "white").lower()
        elif kind.startswith("ball"):
            res.ball = _point(payload.get("ball"), sx, sy, w, h)
        elif kind == "numbers":
            res.numbers = _jc.parse_numbers(payload, scale=(sx, sy))
        else:
            res.landmarks, res.ball = parse_landmarks(payload, sx, sy, w, h)
        return res

    def decide(
        self, images, questions, text: Optional[str] = None, max_width: int = 1920, timeout=60
    ) -> Optional[dict]:
        """OpenAI Decisions API: `questions` ({type: predicate|choice|score, name, instructions,
        ...}) about `images` (BGR, sent inline as JPEG in order) and optional shared `text`.
        Returns {"answers": {name: answer}} (predicate -> "probability"), or None when the budget
        would be exceeded or the call fails. Synchronous; costed against `max_usd`."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        reserve = (2600 * len(images) + 400 + 60 * len(questions)) * DECISIONS_PRICE_IN / 1e6
        with self._lock:
            if self._spent + self._reserved + reserve > self.max_usd:
                return None
            self._reserved += reserve
        t0, usd, out = time.time(), 0.0, None
        try:
            if self._decide_fn is not None:
                r = self._decide_fn(images, questions, text=text, max_width=max_width)
                usd = float(r.get("_usd", reserve)) if r else 0.0
                out = {"answers": dict(r.get("answers") or {})} if r else None
            else:
                out, usd = self._decide_http(images, questions, text, max_width, timeout, reserve)
        except Exception as exc:  # network / API errors: the caller treats it as "no answer"
            out = None
            self.calls.append(
                dict(model=self.decisions_model, kind="decide", error=type(exc).__name__)
            )
        finally:
            with self._lock:
                self._reserved -= reserve
                self._spent += usd
        if out is not None:
            self.calls.append(
                dict(
                    model=self.decisions_model,
                    kind="decide",
                    images=len(images),
                    questions=len(questions),
                    latency_s=round(time.time() - t0, 2),
                    usd=round(usd, 6),
                )
            )
        return out

    def _decide_http(self, images, questions, text, max_width, timeout, reserve):
        import urllib.request

        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        parts = [{"type": "input_text", "text": text}] if text else []
        for img in images:
            h, w = img.shape[:2]
            s = min(1.0, max_width / float(w))
            if s < 1:
                img = cv2.resize(
                    img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA
                )
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()
            parts.append({"type": "input_image", "image_url": url})
        body = {
            "model": self.decisions_model,
            "input": [{"role": "user", "content": parts}],
            "questions": questions,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/decisions",
            data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            r = json.loads(fh.read())
        tokens = (r.get("usage") or {}).get("input_tokens")
        usd = tokens * DECISIONS_PRICE_IN / 1e6 if tokens is not None else reserve
        return {"answers": {a.get("name"): a for a in r.get("answers", [])}}, usd

    def close(self) -> None:
        self._pool.shutdown(wait=True)


class CachedVisionOracle:
    """Replays answers stored by earlier runs and only calls `inner` for keyframes it has not
    seen, so re-processing the same clip does not pay twice. Cache entries are keyed by video,
    frame index, request kind and model, and hold only parsed coordinates and labels."""

    def __init__(self, inner, cache_path: str, video_key: str):
        self.inner = inner
        self.path = cache_path
        self.video_key = video_key
        self._lock = threading.Lock()
        try:
            with open(cache_path) as fh:
                self._cache = json.load(fh)
        except (OSError, ValueError):
            self._cache = {}

    @property
    def spent_usd(self) -> float:
        return self.inner.spent_usd

    @property
    def calls(self):
        return getattr(self.inner, "calls", [])

    def _key(self, frame_idx: int, kind: str) -> str:
        if hasattr(self.inner, "model_for"):
            model = self.inner.model_for(kind)
        else:
            model = self.inner.people_model if kind == "people" else self.inner.landmark_model
        return f"{self.video_key}|{frame_idx}|{kind}|{model}"

    def memo_get(self, name: str):
        """A small per-video note kept with the cache (e.g. how many windows of this video left
        the field unconfirmed), or None."""
        with self._lock:
            return self._cache.get(f"{self.video_key}|memo|{name}")

    def memo_set(self, name: str, value) -> None:
        with self._lock:
            self._cache[f"{self.video_key}|memo|{name}"] = value

    def submit_people(self, frame_idx: int, frame: np.ndarray) -> Optional[Future]:
        return self._submit(frame_idx, "people", lambda: self.inner.submit_people(frame_idx, frame))

    def submit_landmarks(self, frame_idx: int, frame: np.ndarray) -> Optional[Future]:
        return self._submit(
            frame_idx, "landmarks", lambda: self.inner.submit_landmarks(frame_idx, frame)
        )

    def submit_markings(
        self, frame_idx: int, frame: np.ndarray, alt: bool = False
    ) -> Optional[Future]:
        kind = "markings_alt" if alt else "markings"
        return self._submit(
            frame_idx, kind, lambda: self.inner.submit_markings(frame_idx, frame, alt=alt)
        )

    def submit_ball(self, frame_idx: int, image: np.ndarray, tag: str = "") -> Optional[Future]:
        return self._submit(
            frame_idx, "ball" + tag, lambda: self.inner.submit_ball(frame_idx, image, tag)
        )

    def submit_numbers(self, frame_idx: int, frame: np.ndarray) -> Optional[Future]:
        if not hasattr(self.inner, "submit_numbers"):
            return None
        return self._submit(
            frame_idx, "numbers", lambda: self.inner.submit_numbers(frame_idx, frame)
        )

    def decide(
        self, images, questions, text: Optional[str] = None, max_width: int = 1920, timeout=60
    ) -> Optional[dict]:
        """`inner.decide`, replayed from the cache when the same images and questions were asked
        before (key: SHA-256 of the questions, text, width and image pixels)."""
        import hashlib

        if not hasattr(self.inner, "decide"):
            return None
        imgs = list(images) if isinstance(images, (list, tuple)) else [images]
        h = hashlib.sha256(json.dumps([questions, text, max_width]).encode())
        for img in imgs:
            h.update(str(img.shape).encode())
            h.update(np.ascontiguousarray(img).tobytes())
        key = f"{self.video_key}|decide|{h.hexdigest()[:32]}"
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return {"answers": hit}
        r = self.inner.decide(imgs, questions, text=text, max_width=max_width, timeout=timeout)
        if r is not None:
            # probabilities / labels only (no images, no keys)
            with self._lock:
                self._cache[key] = r["answers"]
        return r

    def _submit(self, frame_idx, kind, call) -> Optional[Future]:
        key = self._key(frame_idx, kind)
        if key in self._cache:
            done: Future = Future()
            done.set_result(self._decode(frame_idx, kind, self._cache[key]))
            return done
        job = call()
        if job is None:
            return None
        out: Future = Future()

        def store(f):
            r = f.result()
            if r is not None:
                with self._lock:
                    self._cache[key] = self._encode(r)
            out.set_result(r)

        job.add_done_callback(store)
        return out

    @staticmethod
    def _encode(r: OracleResult) -> dict:
        return dict(
            people=[dict(box=list(p.box), role=p.role, team=p.team) for p in r.people],
            ball=list(r.ball) if r.ball else None,
            landmarks={k: list(v) for k, v in r.landmarks.items()},
            markings=[list(m) for m in r.markings],
            line_colour=r.line_colour,
            numbers=[dict(n) for n in r.numbers],
            model=r.model,
        )

    @staticmethod
    def _decode(frame_idx: int, kind: str, d: dict) -> OracleResult:
        return OracleResult(
            frame_idx,
            kind,
            people=[OraclePerson(tuple(p["box"]), p["role"], p["team"]) for p in d["people"]],
            ball=tuple(d["ball"]) if d.get("ball") else None,
            landmarks={k: tuple(v) for k, v in d["landmarks"].items()},
            markings=[
                (str(m[0]), float(m[1]), float(m[2]), float(m[3])) for m in d.get("markings", [])
            ],
            line_colour=d.get("line_colour"),
            numbers=[dict(n) for n in d.get("numbers", [])],
            model=d.get("model", ""),
        )

    def close(self) -> None:
        self.inner.close()
        with self._lock:
            with open(self.path, "w") as fh:
                json.dump(self._cache, fh)
