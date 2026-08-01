/**
 * Default Card Renderer
 *
 * What a card looks like when the consumer registers no provider: the node's
 * label plus the three columns core already knows about — tag, weight and
 * depth. Plain DOM, no framework, no stylesheet injection (styles are inline,
 * so a strict CSP and a shadow root both work), and no network access of any
 * kind, since `contentRef` is opaque to core.
 *
 * It doubles as the reference implementation of the provider contract: it
 * owns every child it creates and removes them in `release`, because core
 * does not.
 *
 * @module
 */

import { CARD_DRAG_HANDLE_ATTRIBUTE } from "./overlay.ts";
import type { CardChange, CardNode, CardProvider } from "./types.ts";

/** Everything the default renderer keeps between `mount` and `release`. */
export interface DefaultCardState {
  /** The renderer's own root — the single child it adds to the container. */
  readonly root: HTMLElement;
  readonly label: HTMLElement;
  readonly tag: HTMLElement;
  readonly weight: HTMLElement;
  readonly depth: HTMLElement;
  selected: boolean;
  hovered: boolean;
}

const SURFACE = "rgba(18, 18, 24, 0.92)";
const BORDER_IDLE = "rgba(255, 255, 255, 0.12)";
const BORDER_HOVER = "rgba(255, 255, 255, 0.32)";
const BORDER_SELECTED = "rgba(120, 170, 255, 0.85)";
const TEXT_PRIMARY = "rgba(244, 244, 248, 0.96)";
const TEXT_MUTED = "rgba(244, 244, 248, 0.56)";

function createRoot(document: Document): HTMLElement {
  const root = document.createElement("div");
  Object.assign(root.style, {
    boxSizing: "border-box",
    width: "100%",
    height: "100%",
    display: "flex",
    flexDirection: "column",
    gap: "6px",
    padding: "10px 12px",
    borderRadius: "8px",
    // Longhand: `applyBorder` writes borderColor, and the shorthand would
    // leave it unset until the first selection or hover change.
    borderWidth: "1px",
    borderStyle: "solid",
    borderColor: BORDER_IDLE,
    background: SURFACE,
    color: TEXT_PRIMARY,
    font: "13px/1.35 ui-sans-serif, system-ui, sans-serif",
    overflow: "hidden",
  });
  return root;
}

/**
 * The title row, which doubles as the card's drag handle.
 *
 * The title is the one part of a card whose text nobody wants to select, and a
 * title bar is the conventional place to pick a panel up by — so the readouts
 * below stay selectable while the card as a whole is draggable.
 */
function createLabel(document: Document): HTMLElement {
  const label = document.createElement("div");
  label.setAttribute(CARD_DRAG_HANDLE_ATTRIBUTE, "");
  Object.assign(label.style, {
    fontWeight: "600",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
    cursor: "grab",
    touchAction: "none",
  });
  return label;
}

function createReadouts(document: Document): HTMLElement {
  const readouts = document.createElement("dl");
  Object.assign(readouts.style, {
    display: "grid",
    gridTemplateColumns: "auto 1fr",
    columnGap: "8px",
    rowGap: "2px",
    margin: "0",
    fontSize: "11px",
    color: TEXT_MUTED,
  });
  return readouts;
}

/** Append a `<dt>`/`<dd>` pair to `readouts` and return the value element. */
function appendReadout(readouts: HTMLElement, name: string): HTMLElement {
  const document = readouts.ownerDocument;
  const term = document.createElement("dt");
  term.textContent = name;
  const value = document.createElement("dd");
  value.style.margin = "0";
  value.style.fontVariantNumeric = "tabular-nums";
  readouts.appendChild(term);
  readouts.appendChild(value);
  return value;
}

function writeReadouts(node: CardNode, state: DefaultCardState): void {
  state.label.textContent = node.label ?? String(node.externalId);
  state.tag.textContent = String(node.tag);
  state.weight.textContent = node.weight.toFixed(2);
  state.depth.textContent = String(node.depth);
}

/** Selection outranks hover, so the two states compose instead of clobbering. */
function applyBorder(state: DefaultCardState): void {
  state.root.style.borderColor = state.selected
    ? BORDER_SELECTED
    : state.hovered
    ? BORDER_HOVER
    : BORDER_IDLE;
}

/**
 * Build the provider used when the consumer registers none.
 *
 * A fresh provider per call: it holds no state, but handing out a shared
 * instance would invite consumers to treat it as a singleton to patch.
 */
export function createDefaultCardProvider(): CardProvider<DefaultCardState> {
  return {
    mount(container: HTMLElement, node: CardNode): DefaultCardState {
      const document = container.ownerDocument;
      const root = createRoot(document);
      const label = createLabel(document);
      const readouts = createReadouts(document);
      const state: DefaultCardState = {
        root,
        label,
        tag: appendReadout(readouts, "tag"),
        weight: appendReadout(readouts, "weight"),
        depth: appendReadout(readouts, "depth"),
        selected: false,
        hovered: false,
      };

      root.appendChild(label);
      root.appendChild(readouts);
      writeReadouts(node, state);
      container.appendChild(root);
      return state;
    },

    update(
      _container: HTMLElement,
      node: CardNode,
      change: CardChange,
      state: DefaultCardState,
    ): void {
      switch (change.kind) {
        case "data":
          writeReadouts(node, state);
          return;
        case "selection":
          state.selected = change.selected;
          applyBorder(state);
          return;
        case "hover":
          state.hovered = change.hovered;
          applyBorder(state);
          return;
        case "position":
        case "size":
          // Core already moved and resized the container, and the card's own
          // layout is expressed relative to it.
          return;
      }
    },

    release(_container: HTMLElement, _node: CardNode, state: DefaultCardState): void {
      state.root.remove();
    },
  };
}
