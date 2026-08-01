/**
 * The DOM card renderer.
 *
 * Core decides which nodes are carded, where each card sits and how big it is;
 * everything inside the box is this file's. A card shows what a code-graph user
 * zoomed in this far actually wants: the full path as selectable text, what
 * kind of node it is, and how much is inside it.
 *
 * The title row is the drag handle, so the path below it stays selectable while
 * the card as a whole can be picked up — and dragging it is the same gesture,
 * through the same controller, as dragging the node's sprite on the canvas.
 *
 * @module
 */

import {
  CARD_DRAG_HANDLE_ATTRIBUTE,
  type CardChange,
  type CardNode,
  type CardProvider,
} from "../../../packages/core/mod.ts";
import { KIND_NAMES, type RepoGraph } from "./repo.ts";

/** What the renderer keeps between `mount` and `release`. */
interface RepoCardState {
  /** The single child this renderer adds to the pooled container. */
  readonly root: HTMLElement;
  readonly title: HTMLElement;
  readonly path: HTMLElement;
  readonly stats: HTMLElement;
}

/** A card renderer that also reports how many cards it currently has mounted. */
export interface RepoCardProvider extends CardProvider<RepoCardState> {
  /** Cards currently mounted. The HUD's only source for the figure. */
  mountedCount(): number;
}

function element(tag: string, className: string): HTMLElement {
  const node = document.createElement(tag);
  node.className = className;
  return node;
}

function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`;
}

/**
 * Build a card renderer over a generated repository.
 *
 * Producer columns are read by GPU slot. That is safe here and would not be in
 * general: slots are recycled when nodes are removed, and this demo loads a
 * dataset and never mutates it. A consumer that mutates must key its own
 * records on `node.externalId` instead.
 */
export function createRepoCardProvider(repo: RepoGraph): RepoCardProvider {
  let mounted = 0;

  function describe(node: CardNode, state: RepoCardState): void {
    const slot = node.id;
    const children = repo.childCount[slot];

    state.title.textContent = repo.name[slot];
    state.path.textContent = repo.path[slot];

    const facts = [KIND_NAMES[repo.kind[slot]], `depth ${node.depth}`];
    if (children > 0) {
      facts.push(plural(children, "child", "children"));
      facts.push(`${repo.hierarchy.subtreeSize[slot]} in subtree`);
    }
    state.stats.textContent = facts.join(" · ");
  }

  return {
    mountedCount: () => mounted,

    mount(container: HTMLElement, node: CardNode): RepoCardState {
      const root = element("div", "card");
      const title = element("div", "card-title");
      // Declaring the handle is what makes the card draggable at all; without
      // it a card is a static panel and its text stays trivially selectable.
      title.setAttribute(CARD_DRAG_HANDLE_ATTRIBUTE, "");
      const path = element("div", "card-path");
      const stats = element("div", "card-stats");

      root.append(title, path, stats);
      container.append(root);

      const state: RepoCardState = { root, title, path, stats };
      describe(node, state);
      mounted++;
      return state;
    },

    update(
      _container: HTMLElement,
      node: CardNode,
      change: CardChange,
      state: RepoCardState,
    ): void {
      switch (change.kind) {
        case "data":
          describe(node, state);
          break;
        case "selection":
          state.root.classList.toggle("is-selected", change.selected);
          break;
        case "hover":
          state.root.classList.toggle("is-hovered", change.hovered);
          break;
        // Position and size are applied by core to the container; the card's
        // own layout is percentage-based and needs no work.
        case "position":
        case "size":
          break;
      }
    },

    release(_container: HTMLElement, _node: CardNode, state: RepoCardState): void {
      // Containers are pooled and reused, so removing what this renderer added
      // is this renderer's job.
      state.root.remove();
      mounted--;
    },
  };
}
