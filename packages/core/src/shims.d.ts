/** WGSL shader imports — bare .wgsl and Vite-style ?raw suffix */
declare module "*.wgsl" {
  const source: string;
  export default source;
}

declare module "*.wgsl?raw" {
  const source: string;
  export default source;
}

/** V8-specific Error.captureStackTrace */
interface ErrorConstructor {
  captureStackTrace?(targetObject: object, constructorOpt?: Function): void;
}

/** WebGPU timestamp query extension (not yet in base @webgpu/types) */
interface GPUCommandEncoder {
  writeTimestamp(querySet: GPUQuerySet, queryIndex: number): void;
}
