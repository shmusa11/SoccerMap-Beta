/** Passage bounds use original source indices; playback begins at zero. */
export function passageWindow(clip, start, length) {
  if (
    !Number.isInteger(start) ||
    start < 0 ||
    !Number.isInteger(length) ||
    length < 1
  )
    return null;
  const count = clip?.frames;
  if (count > 0 && start >= count) return null;
  const end = count > 0 ? Math.min(count, start + length) : start + length;
  const fps = clip?.fps || 30;
  return {
    start,
    end,
    seconds: (end - start) / fps,
    sourceSeconds: start / fps,
  };
}

export function runAction(run) {
  if (run.state === "complete") return "Open analysis";
  if (run.state === "running") return "View progress";
  return run.resumable ? "Review & resume" : "View details";
}
