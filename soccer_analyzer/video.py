"""Video resource management shared by the app and offline processing."""

import queue
import threading
from contextlib import contextmanager
from math import isfinite
from typing import Callable, Iterator, Optional, Tuple

import cv2


def frame_limit(reported_count: float, max_frames: int | None) -> int:
    """Bound decoding even when a container does not report its frame count.

    Zero means read until EOF, and is returned only when neither limit is known.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    count = int(reported_count) if isfinite(reported_count) and reported_count > 0 else 0
    if max_frames is None:
        return count
    return min(count, max_frames) if count else max_frames


def sampled_fps(source_fps: float, stride: int = 1) -> float:
    """Preserve source timing when retaining every Nth frame; unknown FPS uses 30."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    fps = source_fps if isfinite(source_fps) and source_fps > 0 else 30.0
    return fps / stride


def frame_window(
    reported_count: float, max_frames: int | None, start_frame: int = 0
) -> tuple[int, int | None]:
    """Return [start, stop) in original frames; None stop means unknown-length EOF.

    ``max_frames`` counts frames in the requested window, not an absolute end index.
    A known out-of-range start is rejected before any model loads.
    """
    if type(start_frame) is not int or start_frame < 0:
        raise ValueError("The starting frame must be a nonnegative integer.")
    count = frame_limit(reported_count, None)
    if count and start_frame >= count:
        raise ValueError("The starting frame is past the end of this video.")
    if max_frames is not None:
        frame_limit(0, max_frames)
        stop = start_frame + max_frames
        return start_frame, min(count, stop) if count else stop
    return start_frame, count or None


def seek_capture(cap, index: int, *, fresh: bool = True) -> None:
    """Seek to an exact raw index or fail instead of mislabelling frames.

    A fresh capture already starts at zero. Pass ``fresh=False`` when reusing a reader,
    including when rewinding to zero.
    """
    if type(index) is not int or index < 0:
        raise ValueError("The frame index must be a nonnegative integer.")
    if index == 0 and fresh:
        return
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
        raise ValueError(
            "This video cannot seek to that frame. Export the passage as a separate clip."
        )
    actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if not isfinite(actual) or abs(actual - index) > 0.5:
        raise ValueError(
            "The video reader could not seek accurately. Export this passage as a separate clip."
        )


@contextmanager
def open_capture(path):
    """Open a video and release it on success, failure, or cancellation."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        yield cap
    finally:
        cap.release()


def open_writer(path, fps, size):
    """Try browser-friendly H.264, then MPEG-4; never silently write a broken file."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("No supported MP4 encoder is available (tried avc1 and mp4v).")


def skip_frames(cap, n: int) -> int:
    """Advance `cap` by `n` frames with sequential grab() calls; returns how many were skipped.

    Never seek with cap.set(cv2.CAP_PROP_POS_FRAMES, i) when frame indices matter: on HEVC phone
    clips and 4K H.264 it lands on the wrong frame while reporting the requested position
    (measured 2026-10-12: a seek to 352 returned true frame 372), so every later index would be
    silently off. Counting grabs from the start is the only index that is always right.
    """
    if n < 0:
        raise ValueError("cannot skip a negative number of frames")
    k = 0
    while k < n and cap.grab():
        k += 1
    return k


_END = object()


@contextmanager
def prefetch_frames(
    cap,
    first: int,
    stop: Optional[int],
    want: Callable[[int], bool],
    depth: int = 6,
    extra: Optional[Callable] = None,
) -> Iterator[Iterator[Tuple[int, object, object]]]:
    """Sequential decode on a background thread, so decoding overlaps the caller's work.

    Yields an iterator of (idx, frame, aux) for every frame from `first` (the index of the next
    frame `cap` returns) up to `stop` (exclusive; None or 0 = end of video), in order, never
    seeking (see `skip_frames`). `frame` is the BGR image when `want(idx)`, else None (grab
    only); `aux = extra(frame)` is computed on the decoding thread for retrieved frames. The
    frames are exactly those of a grab()/retrieve() loop, which stops at the first failure. At
    most `depth` frames wait in the buffer. The thread is stopped and joined on exit, before
    `cap` can be released."""
    buf: "queue.Queue" = queue.Queue(maxsize=max(1, int(depth)))
    halt = threading.Event()

    def put(item) -> bool:
        while not halt.is_set():
            try:
                buf.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def work():
        idx = first
        try:
            while not stop or idx < stop:
                if halt.is_set() or not cap.grab():
                    break
                frame = aux = None
                if want(idx):
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    if extra is not None:
                        aux = extra(frame)
                if not put((idx, frame, aux)):
                    return
                idx += 1
        except BaseException as e:  # re-raised in the consumer
            put(e)
            return
        put(_END)

    def frames():
        while True:
            item = buf.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    t = threading.Thread(target=work, name="frame-prefetch", daemon=True)
    t.start()
    try:
        yield frames()
    finally:
        halt.set()
        t.join()
