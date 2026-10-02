// The few Node built-ins the *.test.ts files use, typed just enough for
// `tsc --noEmit` (the workbench itself never imports them).

declare module "node:test" {
  type Body = () => void | Promise<void>;
  export default function test(name: string, body: Body): Promise<void>;
}

declare module "node:assert/strict" {
  const assert: {
    (value: unknown, message?: string): asserts value;
    ok(value: unknown, message?: string): asserts value;
    fail(message?: string): never;
    equal(actual: unknown, expected: unknown, message?: string): void;
    notEqual(actual: unknown, expected: unknown, message?: string): void;
    deepEqual(actual: unknown, expected: unknown, message?: string): void;
    match(value: string, pattern: RegExp, message?: string): void;
    throws(block: () => unknown, expected?: RegExp | object, message?: string): void;
    rejects(block: Promise<unknown> | (() => Promise<unknown>), expected?: RegExp | object, message?: string): Promise<void>;
  };
  export default assert;
}

declare module "node:fs" {
  export function readFileSync(path: string | URL, encoding: "utf8"): string;
  export function readdirSync(path: string | URL, options?: { recursive?: boolean }): string[];
}

declare module "node:vm" {
  export function runInNewContext(code: string, context: Record<string, unknown>): unknown;
}
