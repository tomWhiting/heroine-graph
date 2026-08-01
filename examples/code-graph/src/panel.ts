/**
 * The control panel and the HUD readout.
 *
 * Pure DOM plumbing over {@link KnobState}. Every control edits one field and
 * hands the whole next state back; nothing here knows what a threshold or a
 * budget means, which is {@link knobs} module's job.
 *
 * @module
 */

import { formatCount, formatMs, type HudSnapshot } from "./hud.ts";
import { KNOB_RANGES, type KnobState } from "./knobs.ts";
import { REPO_SCALES } from "./repo.ts";

/** What the panel reports back to the application. */
export interface PanelHandlers {
  /** A knob moved; `next` is the complete new state. */
  onChange(next: KnobState): void;
  /**
   * The reheat button was pressed.
   *
   * The only place in this demo that raises simulation alpha. Reheating is a
   * layout decision, and the user is the only one entitled to make it — no
   * camera move, LOD transition or card mount reaches this.
   */
  onReheat(): void;
}

/** Handle on a mounted panel. */
export interface Panel {
  /** Disable the controls while a dataset is being generated and uploaded. */
  setBusy(busy: boolean): void;
}

function labelled(text: string, control: HTMLElement, hint?: string): HTMLElement {
  const row = document.createElement("div");
  row.className = "knob";
  const label = document.createElement("label");
  label.className = "knob-label";
  label.textContent = text;
  row.append(label, control);
  if (hint !== undefined) {
    const note = document.createElement("p");
    note.className = "knob-hint";
    note.textContent = hint;
    row.append(note);
  }
  return row;
}

/**
 * Mount the five controls.
 *
 * The panel holds the authoritative knob state: every control writes its field
 * into it and emits the whole thing, so no caller has to reassemble a partial
 * update.
 */
export function mountPanel(
  root: HTMLElement,
  initial: KnobState,
  handlers: PanelHandlers,
): Panel {
  let state = initial;
  const inputs: (HTMLInputElement | HTMLSelectElement | HTMLButtonElement)[] = [];

  function emit(patch: Partial<KnobState>): void {
    state = { ...state, ...patch };
    handlers.onChange(state);
  }

  // --- 1. Scale ---------------------------------------------------------
  const scale = document.createElement("select");
  scale.className = "knob-select";
  for (const option of REPO_SCALES) {
    const item = document.createElement("option");
    item.value = option.id;
    item.textContent = `${option.label} — ${option.note}`;
    scale.append(item);
  }
  scale.value = state.scale;
  scale.addEventListener("change", () => {
    const chosen = REPO_SCALES.find((option) => option.id === scale.value);
    if (chosen) emit({ scale: chosen.id });
  });
  inputs.push(scale);
  root.append(labelled("Repository size", scale));

  // --- 2. Level of detail ----------------------------------------------
  const detailRow = document.createElement("div");
  detailRow.className = "knob-inline";

  const detail = document.createElement("input");
  detail.type = "range";
  detail.min = String(KNOB_RANGES.detail.min);
  detail.max = String(KNOB_RANGES.detail.max);
  detail.step = String(KNOB_RANGES.detail.step);
  detail.value = String(state.detail);
  detail.disabled = !state.lodEnabled;
  detail.addEventListener("input", () => emit({ detail: Number(detail.value) }));

  const lodEnabled = document.createElement("input");
  lodEnabled.type = "checkbox";
  lodEnabled.checked = state.lodEnabled;
  lodEnabled.addEventListener("change", () => {
    detail.disabled = !lodEnabled.checked;
    emit({ lodEnabled: lodEnabled.checked });
  });

  detailRow.append(lodEnabled, detail);
  inputs.push(lodEnabled, detail);
  root.append(
    labelled(
      "Semantic LOD — coarse to fine",
      detailRow,
      "Moves the expand and collapse thresholds together, keeping the hysteresis band.",
    ),
  );

  // --- 3. Card budget ---------------------------------------------------
  const cards = document.createElement("input");
  cards.type = "range";
  cards.min = String(KNOB_RANGES.cardBudget.min);
  cards.max = String(KNOB_RANGES.cardBudget.max);
  cards.step = String(KNOB_RANGES.cardBudget.step);
  cards.value = String(state.cardBudget);
  const cardsValue = document.createElement("output");
  cardsValue.className = "knob-value";
  cardsValue.textContent = String(state.cardBudget);
  cards.addEventListener("input", () => {
    cardsValue.textContent = cards.value;
    emit({ cardBudget: Number(cards.value) });
  });
  const cardsRow = document.createElement("div");
  cardsRow.className = "knob-inline";
  cardsRow.append(cards, cardsValue);
  inputs.push(cards);
  root.append(labelled("DOM card budget", cardsRow, "Cards mounted at once, at most."));

  // --- 4. Label density -------------------------------------------------
  const labels = document.createElement("input");
  labels.type = "range";
  labels.min = String(KNOB_RANGES.labelDensity.min);
  labels.max = String(KNOB_RANGES.labelDensity.max);
  labels.step = String(KNOB_RANGES.labelDensity.step);
  labels.value = String(state.labelDensity);
  const labelsValue = document.createElement("output");
  labelsValue.className = "knob-value";
  labelsValue.textContent = String(state.labelDensity);
  labels.addEventListener("input", () => {
    labelsValue.textContent = labels.value;
    emit({ labelDensity: Number(labels.value) });
  });
  const labelsRow = document.createElement("div");
  labelsRow.className = "knob-inline";
  labelsRow.append(labels, labelsValue);
  inputs.push(labels);
  root.append(labelled("Label density", labelsRow, "GPU labels drawn at once, at most."));

  // --- 5. Simulation ----------------------------------------------------
  const runRow = document.createElement("div");
  runRow.className = "knob-inline";
  const run = document.createElement("button");
  run.type = "button";
  run.className = "knob-button";
  run.textContent = state.running ? "Pause" : "Resume";
  run.addEventListener("click", () => {
    const running = !state.running;
    run.textContent = running ? "Pause" : "Resume";
    emit({ running });
  });
  const reheat = document.createElement("button");
  reheat.type = "button";
  reheat.className = "knob-button";
  reheat.textContent = "Reheat";
  reheat.addEventListener("click", () => handlers.onReheat());
  runRow.append(run, reheat);
  inputs.push(run, reheat);
  root.append(
    labelled("Simulation", runRow, "Reheat restarts the layout from where it stands."),
  );

  return {
    setBusy(busy: boolean): void {
      for (const input of inputs) input.disabled = busy;
      detail.disabled = busy || !state.lodEnabled;
    },
  };
}

/** Handle on a mounted HUD. */
export interface Hud {
  render(snapshot: HudSnapshot): void;
}

const HUD_FIELDS = [
  ["nodes", "nodes"],
  ["edges", "edges"],
  ["visible", "visible"],
  ["folded", "folded"],
  ["cards", "cards"],
  ["tick", "tick"],
  ["fps", "fps"],
] as const;

/** Mount the readout and return a render function over {@link HudSnapshot}. */
export function mountHud(root: HTMLElement): Hud {
  const values = new Map<string, HTMLElement>();
  for (const [key, caption] of HUD_FIELDS) {
    const cell = document.createElement("div");
    cell.className = "hud-cell";
    const value = document.createElement("span");
    value.className = "hud-value";
    value.textContent = "—";
    const label = document.createElement("span");
    label.className = "hud-caption";
    label.textContent = caption;
    cell.append(value, label);
    root.append(cell);
    values.set(key, value);
  }

  function write(key: string, text: string): void {
    const cell = values.get(key);
    if (cell) cell.textContent = text;
  }

  return {
    render(snapshot: HudSnapshot): void {
      write("nodes", formatCount(snapshot.nodes));
      write("edges", formatCount(snapshot.edges));
      write("visible", formatCount(snapshot.visible));
      write("folded", formatCount(snapshot.folded));
      write("cards", String(snapshot.cards));
      write("tick", formatMs(snapshot.tickMs));
      write("fps", snapshot.fps > 0 ? String(Math.round(snapshot.fps)) : "—");
    },
  };
}
