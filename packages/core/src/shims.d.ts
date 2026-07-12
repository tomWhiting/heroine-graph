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
  captureStackTrace?(
    targetObject: object,
    constructorOpt?: ((...args: never[]) => unknown) | (new (...args: never[]) => unknown),
  ): void;
}
