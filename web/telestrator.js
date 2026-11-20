/**
 * Coach drawing tool. Marks are stored in pitch metres whenever the view has a field map,
 * so one drawing appears on the map, in 3D and on the film. Where the film frame has no map,
 * a mark falls back to normalised film coordinates and is shown on the film only.
 */
export class Telestrator {
  constructor(onChange) {
    this.onChange = onChange;
    this.marks = [];
    this.tool = "arrow";
    this.color = "#ffd84a";
    this.active = false;
    this.draft = null;
  }

  all() {
    return this.draft ? [...this.marks, this.draft] : this.marks;
  }

  /**
   * Make an element drawable. `locate(event)` returns {space: "pitch"|"film", point, aspect?}
   * or null when the pointer is not over a drawable area.
   */
  attach(element, locate) {
    element.addEventListener("pointerdown", (event) => {
      if (!this.active || event.button !== 0) return;
      const hit = locate(event);
      if (!hit) return;
      event.preventDefault();
      element.setPointerCapture(event.pointerId);
      this.draft = {
        tool: this.tool,
        color: this.color,
        space: hit.space,
        aspect: hit.aspect,
        points: [hit.point, hit.point],
      };
      this.onChange();
    });
    element.addEventListener("pointermove", (event) => {
      if (!this.draft) return;
      const hit = locate(event);
      if (!hit || hit.space !== this.draft.space) return;
      if (this.draft.tool === "pen") this.draft.points.push(hit.point);
      else this.draft.points[1] = hit.point;
      this.onChange();
    });
    const finish = () => {
      if (!this.draft) return;
      const [a, b] = [
        this.draft.points[0],
        this.draft.points[this.draft.points.length - 1],
      ];
      const span = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const tiny = this.draft.space === "film" ? span < 0.01 : span < 0.6;
      if (!tiny || this.draft.points.length > 4) this.marks.push(this.draft);
      this.draft = null;
      this.onChange();
    };
    element.addEventListener("pointerup", finish);
    element.addEventListener("pointercancel", finish);
  }

  undo() {
    this.marks.pop();
    this.onChange();
  }

  clear() {
    this.marks = [];
    this.draft = null;
    this.onChange();
  }
}
