/** WGSL shader imports — bundlers load these as text (esbuild `--loader:.wgsl=text`) */
declare module "*.wgsl" {
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
