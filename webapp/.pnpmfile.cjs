// The project compiles with typescript 7 (native compiler), whose npm package
// exposes no JS compiler API, so typescript-eslint cannot load it. Give the
// @typescript-eslint packages a private typescript 5.x for parsing instead of
// resolving the peer to the project's copy. Remove once typescript-eslint
// supports the native compiler.
const PARSER_TYPESCRIPT_RANGE = "^5.9.2";

function readPackage(pkg) {
  const needsJsApi =
    pkg.name === "typescript-eslint" ||
    (pkg.name ?? "").startsWith("@typescript-eslint/");
  if (needsJsApi && pkg.peerDependencies?.typescript) {
    delete pkg.peerDependencies.typescript;
    pkg.dependencies = {
      ...pkg.dependencies,
      typescript: PARSER_TYPESCRIPT_RANGE,
    };
  }
  return pkg;
}

module.exports = { hooks: { readPackage } };
