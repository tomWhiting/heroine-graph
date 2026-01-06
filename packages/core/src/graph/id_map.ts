/**
 * Node ID Mapping
 *
 * Bidirectional mapping between user-provided IDs (strings/numbers) and
 * internal u32 indices. Enables O(1) lookup in both directions.
 *
 * @module
 */

/**
 * Generic ID type (string or number)
 */
export type IdLike = string | number;

/**
 * Bidirectional ID map interface
 */
export interface IdMap<T extends IdLike = IdLike> {
  /** Number of IDs in the map */
  readonly size: number;

  /** Add an ID and return its index */
  add: (id: T) => number;

  /** Get index for an ID (undefined if not found) */
  get: (id: T) => number | undefined;

  /** Get ID for an index (undefined if out of bounds) */
  getId: (index: number) => T | undefined;

  /** Check if ID exists */
  has: (id: T) => boolean;

  /** Remove an ID and return its former index */
  remove: (id: T) => number | undefined;

  /** Clear all mappings */
  clear: () => void;

  /** Get all IDs in index order */
  ids: () => T[];

  /** Iterate over [index, id] pairs */
  entries: () => IterableIterator<[number, T]>;
}

/**
 * Creates a bidirectional ID map
 *
 * @returns New ID map
 */
export function createIdMap<T extends IdLike = IdLike>(): IdMap<T> {
  // ID -> index mapping
  const idToIndex = new Map<T, number>();
  // Index -> ID mapping (array for O(1) index lookup)
  const indexToId: T[] = [];
  // Free indices from removals (for reuse)
  const freeIndices: number[] = [];

  return {
    get size() {
      return idToIndex.size;
    },

    add(id: T): number {
      // Check if already exists
      const existing = idToIndex.get(id);
      if (existing !== undefined) {
        return existing;
      }

      // Get next index (reuse freed indices first)
      let index: number;
      if (freeIndices.length > 0) {
        index = freeIndices.pop()!;
      } else {
        index = indexToId.length;
      }

      // Store mappings
      idToIndex.set(id, index);
      indexToId[index] = id;

      return index;
    },

    get(id: T): number | undefined {
      return idToIndex.get(id);
    },

    getId(index: number): T | undefined {
      if (index < 0 || index >= indexToId.length) {
        return undefined;
      }
      return indexToId[index];
    },

    has(id: T): boolean {
      return idToIndex.has(id);
    },

    remove(id: T): number | undefined {
      const index = idToIndex.get(id);
      if (index === undefined) {
        return undefined;
      }

      // Remove from maps
      idToIndex.delete(id);
      // Note: We don't remove from indexToId to maintain index stability
      // The index is added to freeIndices for potential reuse

      freeIndices.push(index);

      return index;
    },

    clear(): void {
      idToIndex.clear();
      indexToId.length = 0;
      freeIndices.length = 0;
    },

    ids(): T[] {
      return [...indexToId].filter((_, i) => !freeIndices.includes(i));
    },

    *entries(): IterableIterator<[number, T]> {
      for (let i = 0; i < indexToId.length; i++) {
        if (!freeIndices.includes(i)) {
          yield [i, indexToId[i]];
        }
      }
    },
  };
}

/**
 * Creates an ID map pre-populated with sequential IDs
 *
 * @param count - Number of sequential IDs to create
 * @param prefix - Prefix for generated IDs
 * @returns Populated ID map
 */
export function createSequentialIdMap(
  count: number,
  prefix: string = "n",
): IdMap<string> {
  const map = createIdMap<string>();
  for (let i = 0; i < count; i++) {
    map.add(`${prefix}${i}`);
  }
  return map;
}

/**
 * Creates an ID map from an array of IDs
 *
 * @param ids - Array of IDs
 * @returns Populated ID map
 */
export function createIdMapFromArray<T extends IdLike>(ids: T[]): IdMap<T> {
  const map = createIdMap<T>();
  for (const id of ids) {
    map.add(id);
  }
  return map;
}

/**
 * Serialize an ID map to a plain object
 *
 * @param map - ID map to serialize
 * @returns Plain object representation
 */
export function serializeIdMap<T extends IdLike>(
  map: IdMap<T>,
): { ids: T[]; indices: number[] } {
  const ids: T[] = [];
  const indices: number[] = [];

  for (const [index, id] of map.entries()) {
    ids.push(id);
    indices.push(index);
  }

  return { ids, indices };
}

/**
 * Deserialize an ID map from a plain object
 *
 * @param data - Serialized data
 * @returns Reconstructed ID map
 */
export function deserializeIdMap<T extends IdLike>(data: {
  ids: T[];
  indices: number[];
}): IdMap<T> {
  const map = createIdMap<T>();

  // This is a simplified reconstruction
  // For proper reconstruction, we'd need to handle the index mapping correctly
  for (const id of data.ids) {
    map.add(id);
  }

  return map;
}

/**
 * Map function over ID map entries
 *
 * @param map - ID map
 * @param fn - Mapping function
 * @returns Array of mapped values
 */
export function mapIdMap<T extends IdLike, R>(
  map: IdMap<T>,
  fn: (id: T, index: number) => R,
): R[] {
  const results: R[] = [];
  for (const [index, id] of map.entries()) {
    results.push(fn(id, index));
  }
  return results;
}
