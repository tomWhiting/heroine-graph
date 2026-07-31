/// <reference types="vue/jsx-runtime" />

/**
 * Ambient declaration for single-file components.
 *
 * `vue-tsc` understands `.vue` files natively, but plain `tsc` (and editors
 * that fall back to it) needs this shim to resolve `./GraphMother.vue`.
 */
declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<
    Record<string, unknown>,
    Record<string, unknown>,
    unknown
  >;
  export default component;
}
