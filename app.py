"""
Soccer → 2D map: upload a clip, watch it process live.

Run:
    export PNLCALIB_ROOT=/path/to/PnLCalib
    streamlit run app.py

Left = broadcast frame with an estimated pitch projection on calibrated frames + detected players.
The projection includes model geometry beyond directly observed landmarks.
Right = the 2D top-down map: the full ESTIMATED field + players + ball + camera footprint,
with green shading showing which parts of the pitch have been measured over the clip.
"""

import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

from dotenv import load_dotenv

import cv2
import numpy as np
import streamlit as st

from soccer_analyzer.calibration.pitch_model import PitchModel
from soccer_analyzer.calibration.field_detector import PnLCalibFieldDetector
from soccer_analyzer.calibration.mapper import FrameCalibration
from soccer_analyzer.calibration.lookup import CalibrationLookup
from soccer_analyzer.video import frame_limit, open_capture, open_writer, sampled_fps
from soccer_analyzer.calibration.clip_calibrator import calibrate_clip_auto
from soccer_analyzer.calibration.fusion import CoverageMap
from soccer_analyzer.calibration.visualize import draw_calibration_overlay, draw_topdown, compose

load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title="Soccer → 2D map", layout="wide")
st.title("⚽ Soccer → 2D top-down map")
st.caption(
    "Upload a game clip → automatic field calibration + player tracking → live 2D map. "
    "No manual setup, no wearables."
)

# fixed marker colours: team A red, team B blue, referee yellow, goalkeeper magenta, unknown grey
TEAM_BGR = {
    0: (40, 40, 235),
    1: (235, 120, 40),
    -1: (170, 170, 170),
    -2: (0, 255, 255),
    -3: (255, 0, 255),
}


with st.sidebar:
    st.header("Settings")
    root = st.text_input(
        "PnLCalib repo path",
        os.environ.get("PNLCALIB_ROOT", str(Path(__file__).resolve().parent.parent / "PnLCalib")),
    )
    st.subheader("Field size (varies by level)")
    length = st.number_input(
        "Length (m)",
        90.0,
        120.0,
        105.0,
        0.5,
        help="FIFA 105. Youth/HS/college vary. The fixed markings "
        "(circle, boxes) set the scale; this sets the boundary.",
    )
    width = st.number_input("Width (m)", 45.0, 90.0, 68.0, 0.5)
    st.subheader("Pipeline")
    engine = st.radio(
        "Engine",
        ["Fast keyframe pipeline", "Classic three-pass"],
        help="Fast: decode once, calibrate only when the camera moves, track players at ~7 Hz, "
        "optional AI keyframes for teams/roles/on-field filtering. Classic: the original "
        "pipeline (calibration on every 4th frame).",
    )
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    oracle_on = st.checkbox(
        "AI keyframes (OpenAI vision model)",
        has_key,
        disabled=not has_key,
        help="Every ~2 s a vision model labels players, teams, referees and keepers (and, with "
        "the options below, names the field markings and checks the ball). Needs "
        "OPENAI_API_KEY; sends sampled frames and crops to OpenAI.",
    )
    oracle_usd = st.number_input("AI budget per clip (USD)", 0.05, 5.0, 0.30, 0.05)
    markings_on = st.checkbox(
        "AI field markings (fast pipeline calibration)",
        has_key and oracle_on,
        disabled=not (has_key and oracle_on),
        help="Every ~2 s the stronger model names the visible soccer markings. A camera is fitted "
        "only when two keyframes agree, then carried through the camera motion and checked "
        "against the painted lines on every frame; frames that fail show no field. Fixes night "
        "and multi-sport fields; about $0.08 per 10 s of video.",
    )
    ball_checks_on = st.checkbox(
        "AI ball checks (fast pipeline)",
        has_key and oracle_on,
        disabled=not (has_key and oracle_on),
        help="About once a second the cheap model checks the ball track around its current "
        "position (skipped at night, where it is unreliable). About $0.0003 per check.",
    )
    players_on = st.checkbox("Detect players + ball (local YOLO)", True)
    enhance = st.checkbox(
        "Enhance dark/night footage",
        False,
        help="CLAHE brighten/contrast before detection — helps night games",
    )
    verify_lines = st.checkbox(
        "Re-verify each rendered frame against real lines",
        False,
        help="Extra per-frame check at render time. The calibrator already "
        "gates on line-support + geometry, so this is usually "
        "redundant and costs ~0.1s/frame.",
    )
    line_thr = st.slider("Line-support threshold", 0.0, 0.6, 0.28, 0.02)
    maxf = st.slider(
        "Max frames",
        30,
        7200,
        300,
        30,
        help="Everything runs LOCALLY (no per-frame API cost) — long clips are "
        "safe, they just take proportionally longer.",
    )
    stride = st.slider("Process every Nth frame (higher = faster, longer clips)", 1, 10, 3)
    ball_every = st.slider(
        "Ball detection cadence (higher = faster, less frequent scans)",
        1,
        8,
        2,
        help="The ball is found by WASB tiled detection + an offline trajectory solve "
        "(links candidates over time, keeps the moving ball). Higher = "
        "scan less often, interpolate short gaps.",
    )

up = st.file_uploader("Upload a game clip", type=["mp4", "mov", "avi", "m4v"])
go = st.button("▶ Process", type="primary", disabled=up is None)


def _load_field_detector(root_, length_, width_, enhance_):
    """Reuse weights within this session, without sharing mutable detector settings."""
    key = (root_, length_, width_, enhance_)
    cached = st.session_state.get("field_detector")
    if cached is not None and cached[0] == key:
        return cached[1]
    detector = PnLCalibFieldDetector(
        pnlcalib_root=root_,
        weights_kp=os.path.join(root_, "SV_kp"),
        weights_line=os.path.join(root_, "SV_lines"),
        pitch=PitchModel(length=length_, width=width_),
        enhance=enhance_,
        max_reproj_px=1000.0,
        min_line_support=0.0,
    )
    detector._ensure_loaded()  # Fail during setup, not as silent calibration misses.
    st.session_state["field_detector"] = (key, detector)
    return detector


def _load_player_models():
    if "player_models" in st.session_state:
        return st.session_state["player_models"]
    from soccer_analyzer.detection import SoccerDetector

    pdet_ = SoccerDetector()
    try:
        from soccer_analyzer.ball_wasb import WASBBallDetector

        wasb_ = WASBBallDetector()
    except Exception as exc:
        wasb_ = None
        st.warning(f"Optional ball detector unavailable: {exc}")
    st.session_state["player_models"] = (pdet_, wasb_)
    return pdet_, wasb_


def _setup(path, pitch):
    # auto-enable enhancement only on genuinely dark clips. Sample brightness ACROSS the clip and
    # use the MEDIAN — sampling only the first frames misfires on dark intros/scoreboards (a bright
    # daytime clip then gets CLAHE'd, which pushes the broadcast-trained CNN out of distribution
    # and wrecks calibration). Median over the whole clip is robust to intros and transient dips.
    auto_enh = enhance
    if not auto_enh:
        with open_capture(path) as cap:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            vals = []
            for fi in np.linspace(total * 0.1, total * 0.9, 12).astype(int) if total else range(0):
                # Index-agnostic: only the brightness matters, so a seek that lands on a nearby
                # frame (some codecs do) is harmless here. Never seek where indices matter.
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, f = cap.read()
                if ok:
                    vals.append(float(f.mean()))
        if vals and np.median(vals) < 60:
            auto_enh = True
    det = _load_field_detector(root, pitch.length, pitch.width, auto_enh)
    msgs = [
        f"enhance: {'on' if auto_enh else 'off'}",
        f"line-verify(post): {'on' if verify_lines else 'off'}",
    ]
    # NO fusion: the unified calibrator keeps each frame's own best homography (raw CNN on good
    # footage ~1.5px; keypoint-anchored on night). Fusion re-derives H from a noisy pooled
    # position and DEGRADES the good frames — it caused the offset/"terrible" maps.
    det.fusion = None
    pdet = wasb = None
    if players_on:
        pdet, wasb = _load_player_models()  # cached across runs
        msgs.append(
            "detector: COCO players + " + ("soccer refs/GK" if pdet.has_soccer else "players only")
        )
        msgs.append("ball: WASB (soccer-pretrained)" if wasb else "ball: unavailable")
    return det, pdet, wasb, msgs


def process_fast(path, workspace):
    """Fast keyframe pipeline: one analysis pass + one render pass."""
    from soccer_analyzer.fast_pipeline import FastConfig, FastPipeline

    pitch = PitchModel(length=length, width=width)
    with st.spinner("Loading models…"):
        det = _load_field_detector(root, pitch.length, pitch.width, False)
        pdet, wasb = _load_player_models()
    oracle = None
    if oracle_on:
        from soccer_analyzer.vision_oracle import OpenAIVisionOracle

        oracle = OpenAIVisionOracle(max_usd=float(oracle_usd))
    # Both need the oracle: ignored (and disabled above) when "AI keyframes" is off.
    config = FastConfig(
        field_markings=markings_on and oracle_on, ball_vlm=ball_checks_on and oracle_on
    )
    pipe = FastPipeline(
        det, pdet, wasb=wasb if players_on else None, oracle=oracle, pitch=pitch, config=config
    )
    prog = st.progress(0.0)
    try:
        with st.spinner(
            "Analysing: camera motion, sparse calibration, players, ball, AI keyframes…"
        ):
            result = pipe.run(path, max_frames=maxf, progress=lambda p: prog.progress(p))
        out_path = str(Path(workspace) / "result.mp4")
        with st.spinner("Rendering the side-by-side video…"):
            pipe.render(path, result, out_path)
    finally:
        if oracle is not None:
            oracle.close()
    s = result.stats
    kept = (s.get("field_markings") or {}).get("kept") or []
    how = f"AI markings, {len(kept)} keyframes" if kept else f"{s['anchors']} anchors"
    st.success(
        f"Done: {s['seconds_of_video']} s of video · calibrated {s['calibrated_pct']:.0f}% of "
        f"frames ({how}) · {s['tracks']} tracks, {s['offfield_tracks']} bystanders hidden "
        f"· ball on {s['ball_path_frames']} frames · AI cost ${s['oracle_usd']:.3f}"
    )
    with st.expander("Timing by stage"):
        st.table(
            [{"stage": n, "seconds": round(t, 2), "calls": c} for n, t, c in pipe.timer.table()]
        )
    st.subheader("Result")
    data = Path(out_path).read_bytes()
    st.video(data)
    st.download_button(
        "⬇ Download result video", data, file_name="soccer_2d_map.mp4", mime="video/mp4"
    )


def process_upload(up, workspace, resources):
    suffix = os.path.splitext(up.name)[1] or ".mp4"
    path = str(Path(workspace) / ("input" + suffix))
    Path(path).write_bytes(up.getvalue())
    if engine == "Fast keyframe pipeline":
        try:
            process_fast(path, workspace)
        except Exception as e:
            st.error(f"Fast pipeline failed: {e}")
        return
    with open_capture(path) as cap:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = frame_limit(cap.get(cv2.CAP_PROP_FRAME_COUNT), maxf)
        output_fps = sampled_fps(cap.get(cv2.CAP_PROP_FPS), stride)
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("The uploaded video contains no decodable frames.")
    fh, fw = first_frame.shape[:2]
    total = total or maxf
    pitch = PitchModel(length=length, width=width)

    try:
        with st.spinner("Loading models + calibrating the fixed camera from the clip…"):
            det, pdet, wasb, msgs = _setup(path, pitch)
        st.success("Ready · " + " · ".join(msgs))
    except Exception as e:
        st.error(f"Setup failed: {e}")
        st.stop()

    # PASS 1 — calibrate the whole clip: per keyframe keep the BEST geometrically-valid
    # homography (raw CNN where it's good ~1.5px; keypoint-anchored on night where the CNN fails),
    # verified against the real lines, then propagate across the (mostly static) camera.
    # ADAPTIVE stride: try coarse (fast) first; only if coverage is poor re-run fine — hard clips
    # can hang off a single anchor frame a coarse stride would skip.
    with st.spinner("Pass 1/3 · calibrating the field (CNN + keypoint-anchored, line-verified)…"):
        res, cstats = calibrate_clip_auto(det, path, pitch=pitch, stride=4, max_frames=maxf)
        covpct = cstats["covered"] / max(1, cstats["frames"]) * 100
        if covpct < 30:
            res, cstats = calibrate_clip_auto(det, path, pitch=pitch, stride=2, max_frames=maxf)
            covpct = cstats["covered"] / max(1, cstats["frames"]) * 100
    cal_src = (
        f"{cstats.get('cnn_frames', 0)} CNN + {cstats.get('keypoint_frames', 0)} keypoint anchors"
    )
    if covpct < 10:
        # MULTI-SPORT FALLBACK: colour-agnostic calibration (discovers which painted colour is
        # the soccer pitch — football/lacrosse/soccer multi-line venues).
        try:
            from soccer_analyzer.calibration.multisport import calibrate_multisport

            with st.spinner("Standard calibration failed — trying multi-sport colour discovery…"):
                ms_res, ms_stats = calibrate_multisport(path, pitch, max_frames=maxf, progress=None)
            if ms_res:
                res = ms_res
                covered_ms = sum(h is not None for h in ms_res.values())
                covpct = 100.0 * covered_ms / max(1, len(ms_res))
                cal_src = f"multi-sport (soccer colour: {ms_stats.get('winning_colour', '?')})"
        except ImportError:
            pass
        except Exception as e:
            st.warning(f"Multi-sport calibration failed: {e}")
    calibration = CalibrationLookup(res)
    calibrated_frames, sampled_frames = calibration.coverage(total, stride)
    covpct = 100.0 * calibrated_frames / max(1, sampled_frames)
    st.info(
        f"Field calibration · {cal_src} "
        f"→ **{calibrated_frames}/{sampled_frames} output frames calibrated ({covpct:.0f}%)**"
    )

    cov = CoverageMap(pitch)
    out_path = str(Path(workspace) / "result.mp4")
    writer = None

    prog = st.progress(0.0)
    stat = st.empty()

    # PASS 2 — OFFLINE SWEEP + GLOBAL SOLVES (MatchProcessor): batched detection + tracking over
    # the whole clip, then decide teams/officials/ball ONCE with all the evidence — one immutable
    # team per track, ball = the best space-time path through all candidates. Consistency by
    # construction instead of per-frame guessing.
    md = None
    if pdet is not None:
        from soccer_analyzer.match_processor import MatchProcessor

        stat.markdown(
            "**Pass 2/3 · detecting & tracking (batched) + solving teams/ball globally…**"
        )
        short_clip = (maxf // max(1, stride)) <= 150  # short clip -> lower ball threshold
        mp_ = MatchProcessor(
            pdet,
            wasb=wasb,
            fps=max(1, round(output_fps)),
            ball_every=ball_every,
            yolo_batch=4,
            ball_thresh=0.28 if short_clip else 0.32,
        )

        pitch_filter = None
        if calibrated_frames:
            # on-pitch test from the calibration: keeps bench/subs/spectators OUT of the global
            # two-team colour fit (they collapse the centroids onto crowd colours otherwise).
            def pitch_filter(fidx, feet):
                H = calibration.at(fidx)
                if H is None:
                    return False  # uncalibrated view (zooms/bench pans) ->
                p = cv2.perspectiveTransform(  # don't let it vote on team colours
                    np.array([[feet]], np.float32), H
                ).reshape(-1, 2)[0]
                return -2 <= p[0] <= pitch.length + 2 and -2 <= p[1] <= pitch.width + 2

        md = mp_.sweep(
            path,
            stride=stride,
            max_frames=maxf,
            progress=lambda p: prog.progress(p),
            pitch_filter=pitch_filter,
        )
        s = md.stats
        st.info(
            f"Sweep · {s['processed']} frames, {s['tracks']} tracks · ball candidates on "
            f"{s['ball_frames_with_cands']} frames → trajectory covers "
            f"{s['ball_path_frames']} frames"
        )

    c1, c2 = st.columns([3, 2])
    bc_ph = c1.empty()
    map_ph = c2.empty()

    def tcode_of(tid):
        k = md.kind.get(tid, "player") if md else "player"
        if k == "referee":
            return -2
        if k == "goalkeeper":
            return -3
        return md.team.get(tid, -1) if md else -1

    # PASS 3 — render with everything known.
    cap = resources.enter_context(open_capture(path))
    covered = processed = idx = 0
    while idx < total:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        H = calibration.at(idx)
        good = H is not None and np.all(np.isfinite(H))
        if good and verify_lines:
            from soccer_analyzer.calibration.line_support import line_distance_field, line_support

            df = line_distance_field(frame)
            if df is None or line_support(H, df, pitch, radius=8) < float(line_thr):
                good = False
                H = None
        calib = FrameCalibration(
            idx,
            H if good else np.eye(3),
            1.0 if good else 0.0,
            "ok" if good else "none",
            0,
            0,
            None,
        )

        players_pitch, ball_pitch, boxes = [], None, []
        ball_img = None
        if md is not None:
            people = md.people.get(idx, [])
            b = md.ball.get(idx)
            if b is None:  # nearest ball-path frame (small gap)
                for d in (1, 2):
                    b = md.ball.get(idx - d) or md.ball.get(idx + d)
                    if b:
                        break
            if b is not None:
                ball_img = (b[0], b[1])
            if good and people:
                feet = np.array([p[2] for p in people], np.float32)
                pp = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), H).reshape(-1, 2)
                for i, (tid, bbox, ft) in enumerate(people):
                    x, y = float(pp[i][0]), float(pp[i][1])
                    if -5 <= x <= pitch.length + 5 and -5 <= y <= pitch.width + 5:
                        players_pitch.append((x, y, tcode_of(tid)))
            if good and ball_img is not None:
                bp = cv2.perspectiveTransform(np.array([[ball_img]], np.float32), H).reshape(-1, 2)[
                    0
                ]
                if 0 <= bp[0] <= pitch.length and 0 <= bp[1] <= pitch.width:
                    ball_pitch = (float(bp[0]), float(bp[1]))
            boxes = [(bbox, tcode_of(tid)) for (tid, bbox, ft) in people]
        if good:
            cov.add(H, (fw, fh))
            covered += 1

        overlay = draw_calibration_overlay(frame, calib, pitch, fade_by_confidence=False)
        for bbox, team in boxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), TEAM_BGR.get(team, (170, 170, 170)), 2)
        if ball_img is not None:
            bx, by = int(ball_img[0]), int(ball_img[1])
            cv2.circle(overlay, (bx, by), 13, (255, 255, 255), -1)
            cv2.circle(overlay, (bx, by), 13, (0, 0, 0), 2)
        if not good:
            cv2.putText(
                overlay,
                "NO CALIBRATION HERE",
                (15, fh - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )
        topdown = draw_topdown(
            pitch,
            calib if good else None,
            players_pitch,
            ball_pitch,
            cov.normalized(),
            frame_size=(fw, fh),
        )

        canvas = compose(overlay, topdown, out_width=1280)
        if writer is None:
            writer = open_writer(out_path, output_fps, (canvas.shape[1], canvas.shape[0]))
            resources.callback(writer.release)
        writer.write(canvas)
        if processed % 4 == 0:  # live preview without render drag
            bc_ph.image(overlay[:, :, ::-1], channels="RGB", use_container_width=True)
            map_ph.image(topdown[:, :, ::-1], channels="RGB", use_container_width=True)
        processed += 1
        idx += 1
        prog.progress(min(1.0, idx / total))
        stat.markdown(
            f"**Pass 3/3 · frame {idx}/{total}** · players on pitch: "
            f"**{len(players_pitch)}** · calibration: {'covered' if good else 'none'} "
            f"· covered {covered}/{processed}"
        )
    cap.release()
    if writer is not None:
        writer.release()

    covpct = 100.0 * covered / max(1, processed)
    if covpct < 35:
        st.error(
            f"⚠️ Only {covpct:.0f}% of frames could be calibrated — very hard footage "
            f"(night / wide-angle / multi-sport lines). The field + players are shown only "
            f"where the calibration was verified against the real lines."
        )
    else:
        st.success(
            f"Done — {covered}/{processed} output frames calibrated "
            f"({covpct:.0f}%), field measured {cov.fraction_seen() * 100:.0f}%."
        )

    st.subheader("Result")
    try:
        with open(out_path, "rb") as f:
            data = f.read()
        st.video(data)
        st.download_button(
            "⬇ Download result video", data, file_name="soccer_2d_map.mp4", mime="video/mp4"
        )
    except Exception as e:
        st.warning(f"Could not embed the video ({e}); processing still completed.")


if go and up:
    try:
        with tempfile.TemporaryDirectory(prefix="soccer-map-") as workspace:
            with ExitStack() as resources:
                process_upload(up, workspace, resources)
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
