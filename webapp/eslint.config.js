import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/", "node_modules/"] },
  js.configs.recommended,
  // Deliberately not the type-checked variants: they require a full type-check
  // per run and would make lint too slow for pre-commit use.
  tseslint.configs.recommended,
  // This plugin version keys its flat-format configs under "flat".
  reactHooks.configs.flat.recommended,
  reactRefresh.configs.vite,
  {
    rules: {
      // The codebase marks intentionally-unused bindings with a leading
      // underscore (rest-destructuring omissions, ignored callback params).
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
          destructuredArrayIgnorePattern: "^_",
        },
      ],
      // React Compiler diagnostics: this app does not build with the compiler,
      // and the flagged patterns are deliberate, commented tradeoffs whose
      // "fixes" would change behavior.
      "react-hooks/set-state-in-effect": "off",
      "react-hooks/preserve-manual-memoization": "off",
      "react-hooks/purity": "off",
      "react-hooks/immutability": "off",
    },
  },
);
