# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0](https://github.com/Ynewtime/markitai/compare/v0.15.0...v1.0.0) (2026-07-04)


### ⚠ BREAKING CHANGES

* output names replace the input extension (sample.pdf → sample.md); doctor exits 1 when required deps are missing; mixing INPUT with a subcommand errors instead of silently dropping the input; URLs are no longer sent to remote extraction services without consent.
* release v0.4.0 - module refactoring, LLM providers, security hardening
* release v0.3.2 - Chinese localization, improved setup scripts
* release v0.3.1 - prompt leakage fix, SPA learning, Windows optimizations
* release v0.3.0 - URL conversion, package rename, website
* Complete rewrite with new architecture

### Features

* --pure without --llm writes raw markdown without frontmatter ([b917640](https://github.com/Ynewtime/markitai/commit/b9176403587c3009c6524cfa96b4e0b24ca97b8d))
* add --keep-base CLI option ([9854bc5](https://github.com/Ynewtime/markitai/commit/9854bc5350386bca0587807ff6349ffa14fa58a3))
* add CLAUDE.md & AGENTS.md ([e8e1d8c](https://github.com/Ynewtime/markitai/commit/e8e1d8c50e09755494bf10ef067c1bc81ed1c248))
* add content-addressed asset store with symlink refs ([d4cf378](https://github.com/Ynewtime/markitai/commit/d4cf37897aaaa65a966a07f18b89a8c7852613ab))
* add FxTwitter API client for tweet fetching ([2a48b71](https://github.com/Ynewtime/markitai/commit/2a48b71b38d3f23a30d7a89878e37eba006bfbca))
* add keep_base field to LLMConfig ([e49bcee](https://github.com/Ynewtime/markitai/commit/e49bcee955563d368d102b8e75118203b6bd8311))
* add model deprecation warnings and standardize error messages to English ([9781507](https://github.com/Ynewtime/markitai/commit/978150786b8170318b7405a6dc1469f95b07a281))
* add narrow markdown fidelity layer for canonical content ([3b6f686](https://github.com/Ynewtime/markitai/commit/3b6f68648ff6b6d5b1d3b6650906af3afb2b0609))
* add optional language field to Frontmatter model, strengthen image placeholder prompts ([8d8d020](https://github.com/Ynewtime/markitai/commit/8d8d02028d4053b8090a83aa0e7f1ee7b4fcc82e))
* add Playwright Python as primary browser backend ([96327b4](https://github.com/Ynewtime/markitai/commit/96327b4479afe60f4878a294e28da8a65992ef7b))
* add policy-aware async enrichers to resolver pipeline ([6b22a15](https://github.com/Ynewtime/markitai/commit/6b22a1557fac74df558288f5111ef6064849e53c))
* add process_image_with_vision_pure() for --llm --pure image analysis ([a1c1ce6](https://github.com/Ynewtime/markitai/commit/a1c1ce6e57b46dd5b28d522436bac0caf80d70aa))
* add pure mode config field and --pure CLI flag ([ef5a8c2](https://github.com/Ynewtime/markitai/commit/ef5a8c2f33b040e593e18d03c024c98cf4b3ea7e))
* add resilience features and log optimization (v0.1.2) ([e9e7157](https://github.com/Ynewtime/markitai/commit/e9e7157ef100df9f4864f0aeb4d7f7ad1658fd26))
* add setup scripts, PyPI publishing workflow, and English README ([4d41c8e](https://github.com/Ynewtime/markitai/commit/4d41c8e03bb5f301777fba57978ffb2f7be7db6a))
* add shared semantic model for threaded extractions ([1098ea7](https://github.com/Ynewtime/markitai/commit/1098ea7a84bfd5ffaa8023592cbc570c5d38925a))
* add stdout_persist config fields to ImageConfig ([616a3fa](https://github.com/Ynewtime/markitai/commit/616a3fa52b62382734c64554a3c87ca84166c54f))
* add Steam news extractor for BBCode announcement pages ([8cbf767](https://github.com/Ynewtime/markitai/commit/8cbf7671648c2e869f5342d14d87b934c26ad140))
* add terminal image protocol detection (Kitty/iTerm2) ([6aefe55](https://github.com/Ynewtime/markitai/commit/6aefe55eb62593fa26bd6bb65bfaeff0c0320ae7))
* add thread policy module for threaded page inclusion rules ([277e68e](https://github.com/Ynewtime/markitai/commit/277e68ed5e5ac58174b6fa7f58ed259bef27ebf4))
* add typed native extraction quality profiles ([4ddcef1](https://github.com/Ynewtime/markitai/commit/4ddcef1689d79f12aa9eb75a0180fa953fd74275))
* add youtube page extraction on native resolver contract ([622b8bf](https://github.com/Ynewtime/markitai/commit/622b8bf64dba2b047eabf33ada25e4ae91224d99))
* api_base env: syntax support & setup script China mirror acceleration ([a55e28c](https://github.com/Ynewtime/markitai/commit/a55e28cf047a31b43331a49d1c58f4a6948ff3a6))
* auto-detect LLM providers when no model_list configured ([1559974](https://github.com/Ynewtime/markitai/commit/15599741b2522bc91706c0b2e8d739f153315947))
* **cache:** add persistent LLM cache with dual-layer lookup and vision router optimization ([51825e7](https://github.com/Ynewtime/markitai/commit/51825e7674c3b281a83e6e409a6a6990bd580422))
* **claude-agent:** add prompt caching support for long system prompts ([3e4d19c](https://github.com/Ynewtime/markitai/commit/3e4d19c2f413c6c83a2c6a571f2214407229b1b2))
* **claude-agent:** integrate adaptive timeout calculation ([b0ff32e](https://github.com/Ynewtime/markitai/commit/b0ff32eda0ed3b6d65bf1d4f5a8246492c12855b))
* **cli:** add --kreuzberg flag to force kreuzberg converter for all formats ([cbaf0a6](https://github.com/Ynewtime/markitai/commit/cbaf0a6fe3a8d2e2073865d27649d54968d35712))
* **cli:** add --static flag to force static HTTP fetch with native webextract ([be22d13](https://github.com/Ynewtime/markitai/commit/be22d1334014aff77d41d513b2fbaeb51b3a2c08))
* **cli:** add -I/--interactive flag for guided setup ([a271389](https://github.com/Ynewtime/markitai/commit/a271389661d3bfb07138ab7d201ae73931b222bd))
* **cli:** add i18n module for multi-language support ([7e8f751](https://github.com/Ynewtime/markitai/commit/7e8f75190db062027caa00f6afcc0c6bc492ad2f))
* **cli:** add interactive module with provider detection ([9ac3fcd](https://github.com/Ynewtime/markitai/commit/9ac3fcd7c53557f94f7700a0c1b8564d170aa668))
* **cli:** add output directory prompt in interactive mode ([0a79fd4](https://github.com/Ynewtime/markitai/commit/0a79fd4115d3c602a8ae56b41315a7a0bef75e9e))
* **cli:** add run_interactive function with summary output ([097e881](https://github.com/Ynewtime/markitai/commit/097e88116d1e37759917b84ef645b5f1d5ba4a96))
* **cli:** add unified UI components module ([b412edd](https://github.com/Ynewtime/markitai/commit/b412edd3a7651375e2c42e0453e75df13b47959d))
* **cli:** apply unified UI to cache, config, doctor commands ([16e4b27](https://github.com/Ynewtime/markitai/commit/16e4b272fd728f4ed4793cf3bbe43ab58d0c2f0f))
* **cli:** implement interactive prompts with questionary ([eb5042e](https://github.com/Ynewtime/markitai/commit/eb5042ee98d8a49e5f4cfa371183b0689b92037e))
* **cli:** init Update flow, concurrent detection, login failure cards ([c657253](https://github.com/Ynewtime/markitai/commit/c657253237093731f2ee34f6c5cbae1be286e922))
* **cli:** modernize default models, improve init wizard and interactive setup ([fd5c745](https://github.com/Ynewtime/markitai/commit/fd5c7450afc71b85d4b8e674f186054feb609a22))
* **cli:** stage-aware ASCII progress spinner for long conversions ([15bce2a](https://github.com/Ynewtime/markitai/commit/15bce2a7310124b7078ba45d945b6f8f5670e15d))
* **cli:** unified glyph status cards for auth providers ([62ad4ae](https://github.com/Ynewtime/markitai/commit/62ad4ae68f975a17b8f46ba3437a6c380da02800))
* **cli:** upgrade check-deps to doctor command with auth status ([d243e1f](https://github.com/Ynewtime/markitai/commit/d243e1ff96568e7bc16bcd6165215efb2d63926a))
* **cloudflare:** fix --cloudflare flag, env fallback, rate limiting & docs ([98781d7](https://github.com/Ynewtime/markitai/commit/98781d78638930696837b61e174a05e36f092143))
* complete MarkIt v0.0.1 - document to markdown converter ([028876e](https://github.com/Ynewtime/markitai/commit/028876e133c8c2259012c9b3ed687b8c1a98380e))
* **config-editor:** redesign with prompt_toolkit UI (v0.13.1) ([8f41ded](https://github.com/Ynewtime/markitai/commit/8f41ded0b3a148b8cae0c0fcbbd5fdbf0221c22b))
* **config:** add 4 advanced params to CloudflareConfig ([7016886](https://github.com/Ynewtime/markitai/commit/70168865082b2e9b363703d6a4333fe0022566c9))
* **config:** add interactive edit loop with questionary ([26df23f](https://github.com/Ynewtime/markitai/commit/26df23f732cc9deae8db44652135fb16bf24ad3e))
* **config:** add schema introspection for interactive editor ([ae40d62](https://github.com/Ynewtime/markitai/commit/ae40d62c130f6f78a90206911eee571ace70c172))
* **config:** add section separators and edge case handling ([9f1d4ec](https://github.com/Ynewtime/markitai/commit/9f1d4ecb36375c77eb5c1c3f2c4473f018cd956b))
* **config:** add skip_auto_scroll and reject_resource_patterns to DomainProfileConfig ([bf5a809](https://github.com/Ynewtime/markitai/commit/bf5a80998f9e0202e983c9c22fff20eb42d2c63d))
* **config:** wire up 'markitai config edit' subcommand ([3b950a7](https://github.com/Ynewtime/markitai/commit/3b950a7e14b42f65dd84589d763520c4769eca67))
* context-based LLM usage tracking and concurrent safety improvements ([ef27090](https://github.com/Ynewtime/markitai/commit/ef27090f920f55aa799ee7aa8a78781e4108e8bf))
* **converter:** add Cloudflare Workers AI toMarkdown backend ([e817614](https://github.com/Ynewtime/markitai/commit/e817614cbea81355e6edc54c4fcdce905f8ccab3))
* **converter:** route HTML files through webextract pipeline ([bb44912](https://github.com/Ynewtime/markitai/commit/bb449127f394c6c11a272f46e761b0f56ef5efb6))
* **copilot:** integrate adaptive timeout calculation ([981f094](https://github.com/Ynewtime/markitai/commit/981f0941073622cd2ef797963fd8776b0c0c9b19))
* define IMAGE_ONLY_FORMATS constant for raster image formats ([459281d](https://github.com/Ynewtime/markitai/commit/459281d08895481660daa27bca7838126e0cfedc))
* **doctor:** add --fix flag for auto-installing missing components ([c6220bc](https://github.com/Ynewtime/markitai/commit/c6220bc74bc4e37a07b80396ba47ce0396af7686))
* **doctor:** add ChatGPT authentication status check ([7822049](https://github.com/Ynewtime/markitai/commit/7822049db024c67f85eabc406efd4abadf3c852a))
* **doctor:** add cross-platform installation hints ([2e80f32](https://github.com/Ynewtime/markitai/commit/2e80f3290154c172116e168e217bed3305982a67))
* expand threaded extraction coverage on shared abstractions ([1b2853e](https://github.com/Ynewtime/markitai/commit/1b2853ef9138f895ffb377170840fcdbe89fab4b))
* **fetch:** add CF Markdown for Agents content negotiation ([2f24411](https://github.com/Ynewtime/markitai/commit/2f24411095c3b378e8175f25bef528d4110d6947))
* **fetch:** add Cloudflare as unified cloud backend ([a2c0718](https://github.com/Ynewtime/markitai/commit/a2c07182257fac20636656eba3ca8da2610778d8))
* **fetch:** add content validation gate to all fetch strategies ([45ea6bc](https://github.com/Ynewtime/markitai/commit/45ea6bc43dcc1bf0460f28e5d69c87e3b00adecd))
* **fetch:** add policy and domain profile config with safe defaults ([e14c5da](https://github.com/Ynewtime/markitai/commit/e14c5daf5fcc919c29176664476d912690b4811f))
* **fetch:** add policy engine for strategy ordering ([e495524](https://github.com/Ynewtime/markitai/commit/e495524603ab9fa9dfdd07052e74424bd98c9175))
* **fetch:** implement policy-driven strategy selection, session persistence, and adaptive targeting ([44db034](https://github.com/Ynewtime/markitai/commit/44db03460ec38e965e62240c416edfba0cf60083))
* **fetch:** integrate FxTwitter pre-playwright enrichment in dispatch strategy ([679d4f0](https://github.com/Ynewtime/markitai/commit/679d4f016fc38d16f9dc0c9a3eec529c0c99419d))
* **fetch:** pass 4 advanced params in CF BR payload ([d1f728f](https://github.com/Ynewtime/markitai/commit/d1f728f36ba4f2887c93f10b12771474823d0377))
* **fetch:** propagate skip_auto_scroll and reject_resource_patterns from domain profiles ([a66d2bc](https://github.com/Ynewtime/markitai/commit/a66d2bccafdd29175303d0aee21378659e655004))
* **fetch:** wire advanced params at CF BR call sites ([64c641c](https://github.com/Ynewtime/markitai/commit/64c641c61132f87207366a896fc46ea2f4e72c5e))
* fix single URL stdout mode and add external image inline display ([34c060e](https://github.com/Ynewtime/markitai/commit/34c060e2970527c7f55ecec55f5b8314bd10f23f))
* handle image-only skip in batch processor ([f3e7aff](https://github.com/Ynewtime/markitai/commit/f3e7aff324ce1e398bc988f4424082fdfedc28d6))
* implement Phase 3 performance optimizations (Parallel PDF/Image processing) ([3f3bb1b](https://github.com/Ynewtime/markitai/commit/3f3bb1bf6c5e805e6d3f293c73b8494b7ea21e0f))
* implement pure mode pipeline (clean_document_pure, process_document_pure, process_with_pure_llm) ([61951dd](https://github.com/Ynewtime/markitai/commit/61951ddcc710533ee74c9392f14ed6362621fe2a))
* improve batch progress and quiet verbose url logs ([91b2744](https://github.com/Ynewtime/markitai/commit/91b27446caed8a28674703abaf17dd96843de05c))
* improve JSON parsing and log formatting (v0.1.4) ([78d8820](https://github.com/Ynewtime/markitai/commit/78d88206794ba31d9220c07f9932c937cd8e98fc))
* improve test coverage to 81% and fix provider bugs (v0.1.3) ([36d48c5](https://github.com/Ynewtime/markitai/commit/36d48c5b76518385e4837c915aaa79c1f13e6a3a))
* integrate terminal image display and asset persistence into stdout mode ([af9d7f0](https://github.com/Ynewtime/markitai/commit/af9d7f04a86d9539288b8062daabe61f80cd727c))
* **log:** support MARKITAI_LOG_FORMAT env override ([fc8880d](https://github.com/Ynewtime/markitai/commit/fc8880d2252be2329a73a1ab5767e81f4129979b))
* Major architecture refactoring with service layer and enhanced LLM support (v0.1.1) ([aeb9206](https://github.com/Ynewtime/markitai/commit/aeb9206777f8ee6c8aa4890619032e24434f00ee))
* **markdown:** add footnote reference conversion ([^N]) ([7479491](https://github.com/Ynewtime/markitai/commit/747949148b1c24467df6df1cec21825b415899de))
* **markdown:** add highlight (==) and strikethrough (~~) conversion ([05679f0](https://github.com/Ynewtime/markitai/commit/05679f017fac75385f56b814a68cff7d9889ec1f))
* **markdown:** add math formula conversion (KaTeX, MathJax, MathML) ([d1c7bde](https://github.com/Ynewtime/markitai/commit/d1c7bde525ac86e7b3b4178bef6ce4b0f5536342))
* **markdown:** add WebExtractMarkdownConverter with code-block language detection ([d2e1da9](https://github.com/Ynewtime/markitai/commit/d2e1da914b4b1667c311bef7b1e71b7ad433447d))
* native webextract pipeline, .markitai namespace, provider auth, code quality & security hardening ([3361b01](https://github.com/Ynewtime/markitai/commit/3361b01a4dd9deae4dbf50f001877ccbd97ec72e))
* **packaging:** mkai alias package on PyPI ([38a6631](https://github.com/Ynewtime/markitai/commit/38a66316b444e045e12773920459352fb40eabe1))
* **pipeline:** integrate mobile style pruning before content scoring ([7ed743f](https://github.com/Ynewtime/markitai/commit/7ed743f8159e2c9f3698493155ab31124eab9f29))
* **pipeline:** register WebExtractMarkdownConverter for code-block language detection ([dfba519](https://github.com/Ynewtime/markitai/commit/dfba5199ced7fb86684167564742e23ab48cc0dd))
* **playwright:** add built-in domain profiles for x.com, twitter.com, github.com ([186862c](https://github.com/Ynewtime/markitai/commit/186862cf4fb9ff1c719411ab9fad2df6584b6744))
* **playwright:** add site-specific DOM cleanup selectors for x.com ([db641c5](https://github.com/Ynewtime/markitai/commit/db641c566ac2dc8eaca10a73ddbd2eb7a9985a00))
* **playwright:** expose advanced browser capabilities ([177f404](https://github.com/Ynewtime/markitai/commit/177f404ee0a11b0bbf95fa52683e7fefd9b58131))
* **playwright:** implement skip_auto_scroll and smart wait strategy ([316e389](https://github.com/Ynewtime/markitai/commit/316e38987879a3a93ad4e639710ee8baa424c7d9))
* **preprocess:** add React SSR streaming boundary resolution ([af8510c](https://github.com/Ynewtime/markitai/commit/af8510c2c3f7f28c99b89ef87a8891cfe4ca00ef))
* **providers:** add adaptive timeout calculation based on request complexity ([362a41c](https://github.com/Ynewtime/markitai/commit/362a41c3211170eaf9541bd93c9a8f0ee4baa763))
* **providers:** add authentication manager with status caching ([a6f4c79](https://github.com/Ynewtime/markitai/commit/a6f4c795354b152638e4894e44f302363bbe993b))
* **providers:** add structured error classes for local providers ([3df1ae1](https://github.com/Ynewtime/markitai/commit/3df1ae16166a42bf5fcdb674a89214e055b00874))
* **providers:** add unified JSON mode handler for structured outputs ([676f0a8](https://github.com/Ynewtime/markitai/commit/676f0a8a77d6602ce6c8f2b6d1ee0720fd2f8b0c))
* **providers:** export error classes, auth manager, and timeout utilities ([108b0f5](https://github.com/Ynewtime/markitai/commit/108b0f5232b82efdf751f5caaca46164512ffadd))
* **providers:** export StructuredOutputHandler and clean_control_characters ([8fe6931](https://github.com/Ynewtime/markitai/commit/8fe6931dd082ea3e7ea5a2773c667d2ef3a5d806))
* rebuild x extraction on shared thread semantics ([b0b36cc](https://github.com/Ynewtime/markitai/commit/b0b36cc1f5c5bdda608e34ef5efe31c0b1302a0f))
* Refactor CLI architecture and migrate config format to YAML ([a04a265](https://github.com/Ynewtime/markitai/commit/a04a265a4d9d12f675ed9d2bb8c117dc5ed2262c))
* Refactor LLM configuration and add capability routing with load balancing ([e1b2787](https://github.com/Ynewtime/markitai/commit/e1b27874204677d75be3b883f48e7ebcc24af88b))
* refactor prompt management and simplify cleaner (v0.1.5) ([55486ef](https://github.com/Ynewtime/markitai/commit/55486ef50aa6b1ed99edd4918120f036431892f9))
* Release v0.1.0 - Capability-based model routing and enhanced LLM orchestration ([1e8a4a8](https://github.com/Ynewtime/markitai/commit/1e8a4a83242678fae2fe3b2e6c3956badb1cb1f9))
* release v0.1.6 - fix routing strategy and update docs ([bc7dace](https://github.com/Ynewtime/markitai/commit/bc7daceb96ebb76d457da613fc83c6be9e36aeb1))
* release v0.3.0 - URL conversion, package rename, website ([6fcf150](https://github.com/Ynewtime/markitai/commit/6fcf150802f66d2632e2c18110b96c1f73d95935))
* release v0.3.1 - prompt leakage fix, SPA learning, Windows optimizations ([1ada714](https://github.com/Ynewtime/markitai/commit/1ada714b187dd51ff7601334900986da0cdfd8c8))
* release v0.3.2 - Chinese localization, improved setup scripts ([a4b767f](https://github.com/Ynewtime/markitai/commit/a4b767fc74e0b7bebbbdc580dfe2eeff84a5b86c))
* release v0.4.0 - module refactoring, LLM providers, security hardening ([06dbf9c](https://github.com/Ynewtime/markitai/commit/06dbf9c165092a009b1595fb78f21f28630e0c05))
* **removals:** add hero header and trailing thin section removal ([958ddc8](https://github.com/Ynewtime/markitai/commit/958ddc8400e7c9e17253c2ffdb9c387834711032))
* restructure assets.json and improve logging/compatibility ([75b3797](https://github.com/Ynewtime/markitai/commit/75b3797439a735a9649a3d1b1bbf87eeea45cb16))
* rewrite markit v0.2.0 with monorepo architecture ([9803534](https://github.com/Ynewtime/markitai/commit/9803534e540cd663341dcc33f2eea962575384c5))
* route --llm --pure image inputs to Vision analysis path ([48444d5](https://github.com/Ynewtime/markitai/commit/48444d5781b2e56e57444dfd7a75a8b2fdad85db))
* **scripts:** add auto-detection and spinner for install scripts ([710dfeb](https://github.com/Ynewtime/markitai/commit/710dfeb25312486f4822312774f8aa03eb5a144f))
* **scripts:** add installation functions to setup.sh ([82f4579](https://github.com/Ynewtime/markitai/commit/82f45796d3a18b9a165bfedfd68dd169e2377ac8))
* **scripts:** add installation functions to setup.sh.new ([c03db8f](https://github.com/Ynewtime/markitai/commit/c03db8f3b1dd3bd9ffe307fe4abc4c5a2b446e04))
* **scripts:** add main logic to setup.sh ([9c5f1d4](https://github.com/Ynewtime/markitai/commit/9c5f1d4812d3593b592cbb93ac20c2fbfe14dc23))
* **scripts:** add setup.sh skeleton with i18n and clack UI ([78485dd](https://github.com/Ynewtime/markitai/commit/78485dd3b7cd6aacaa239ea02e138e5a37fb88f9))
* **scripts:** add setup.sh skeleton with i18n and clack UI ([4b49636](https://github.com/Ynewtime/markitai/commit/4b496367218dba58114e350896f54bc4a29d5a81))
* **scripts:** apply clack-style UI to all install scripts ([1ea40f9](https://github.com/Ynewtime/markitai/commit/1ea40f9e9efb089fb4e79f96499e82d7fa9f942c))
* **scripts:** consolidate 10 setup scripts into 2 unified files ([078a004](https://github.com/Ynewtime/markitai/commit/078a004a9de201264b59174ea749a505b2cbdbc8))
* **scripts:** show installation mode at startup ([c7be779](https://github.com/Ynewtime/markitai/commit/c7be779f59afd83452965a6975ca26bf2f4d8b82))
* show skip warning for image files in non-LLM mode ([a721be1](https://github.com/Ynewtime/markitai/commit/a721be1b0bd2dc8a2b973c7b2d4ca80947c5b788))
* skip image-only formats in non-LLM/non-OCR mode (Rule A) ([4df7ab0](https://github.com/Ynewtime/markitai/commit/4df7ab0ed6cdffa879b9297c8b8312f715d9eae8))
* skip writing base .md in LLM mode unless --keep-base (Rule B) ([2c0d34d](https://github.com/Ynewtime/markitai/commit/2c0d34d2a22fe6e4e8fc53d8e15a7e01d672de63))
* split raw html preprocess from browser dom normalization ([3a94e5d](https://github.com/Ynewtime/markitai/commit/3a94e5dd1103bd35b434c6adc4c3f7ef28d7324f))
* store detected_format on ConversionContext ([608aa63](https://github.com/Ynewtime/markitai/commit/608aa6370ddf1ec58768b3cdd8ed5811947bd461))
* use mode-specific rules in cleaner prompt for pure vs standard mode ([1e97a9c](https://github.com/Ynewtime/markitai/commit/1e97a9c04f654a662f4d9eb929180da1f5b1062f))
* v0.15.0 overhaul — dependency refresh, bug-fix sweep, local-first fetch, UX rework ([06a3bc9](https://github.com/Ynewtime/markitai/commit/06a3bc940cd07610aedcae63a66e04e04ebf5b18))
* validate shared thread extraction with github discussions ([a7d7fc3](https://github.com/Ynewtime/markitai/commit/a7d7fc3ff94313dbeacd6c227bd41b8524f38a91))
* **vision:** add language-aware retry and rewrite for image analysis ([5770283](https://github.com/Ynewtime/markitai/commit/5770283c2b512f0a9d881bea0689893c358f6aa6))
* warn when --pure silently overrides --alt/--desc/--screenshot ([58c59c8](https://github.com/Ynewtime/markitai/commit/58c59c806f64f9d77939b5b2c0c77a74c74075fd))
* **webextract:** add CSS [@media](https://github.com/media) mobile style pruning ([3af4998](https://github.com/Ynewtime/markitai/commit/3af4998ddaa0fa82471f43182789ac75783c4dd1))
* **webextract:** add noise removal pipeline (Phase 1, defuddle integration) ([168b24c](https://github.com/Ynewtime/markitai/commit/168b24c57bb6a70cd08667153cfeb49257675af7))
* **webextract:** content patterns, heading anchors, callouts (Phase 3) ([bce04ec](https://github.com/Ynewtime/markitai/commit/bce04ecfb95a35110990aa398c89c89d26d051fb))
* **webextract:** enhanced scoring, standardization, multi-level retry (Phase 2) ([2d6c498](https://github.com/Ynewtime/markitai/commit/2d6c498e3955c8606ad4a78ba3246fbf50bf3a6a))
* **webextract:** srcset optimization, enhanced code language detection ([280a7c1](https://github.com/Ynewtime/markitai/commit/280a7c13f7222da3147c603e3bfa810545a7fae4))
* **webextract:** tweet conversion at defuddle parity ([7b2b55d](https://github.com/Ynewtime/markitai/commit/7b2b55d990ee9183435d27109a53e17617f47555))
* write .md as fallback when LLM processing fails ([6f483ad](https://github.com/Ynewtime/markitai/commit/6f483ada98d3743452331dfeba409bfe8b70f975))


### Bug Fixes

* add debug logging to 5 high-risk silent exception handlers ([b000a7b](https://github.com/Ynewtime/markitai/commit/b000a7bd71cf50058fec2973c6798c7d06f721f9))
* add debug logging to all remaining silent exception handlers ([49c7749](https://github.com/Ynewtime/markitai/commit/49c77492bd83cf72cee424e46f7cac90f73b1666))
* add missing assertion and remove redundant create=True mock ([fd5765a](https://github.com/Ynewtime/markitai/commit/fd5765a6059339d8c04f67e83e46294db84387e4))
* add placeholder REMINDER to vision prompts to reduce LLM drift ([e518124](https://github.com/Ynewtime/markitai/commit/e518124e17ddba104902412e0cad7cdae0d3c08c))
* add Sonnet 4.6 to Copilot pricing, note 1M context GA ([86e7e06](https://github.com/Ynewtime/markitai/commit/86e7e0606663b79f75b08ce7f54d27ba721d6ea3))
* add thread safety to ContentCache and _image_cache ([00d4f14](https://github.com/Ynewtime/markitai/commit/00d4f149b4cf58c7a89f5f41ec25c13e2376c456))
* add thread safety to model cooldown tracking ([140b8c1](https://github.com/Ynewtime/markitai/commit/140b8c19032d8c4cdefc27c41fd1381c155e1a13))
* address all code review findings from parity branch review ([81b92fb](https://github.com/Ynewtime/markitai/commit/81b92fb8a2acb920f217e6ff591445409bf4b5c2))
* address code review — 3 robustness improvements ([0577937](https://github.com/Ynewtime/markitai/commit/05779375333389cd9210baca3cb35fb1014ac34b))
* address code review findings — 5 fixes with tests ([430094d](https://github.com/Ynewtime/markitai/commit/430094dc11b857a806f2ba78781bd394d16ca2e9))
* address full code review findings ([d0d84b1](https://github.com/Ynewtime/markitai/commit/d0d84b1141f4d5acee4f322a68efa3ec8ff30e32))
* address review findings — cache correctness, batch screenshot, workflow ordering, perf ([d9cb311](https://github.com/Ynewtime/markitai/commit/d9cb3111b62ca102096ab76c5c4d1c03d938df41))
* address third-party code review — 7 fixes ([97f9e19](https://github.com/Ynewtime/markitai/commit/97f9e19dffaf80125718ce4a275f2f4b5c164587))
* address third-party review of LLM warning implementation ([2065df6](https://github.com/Ynewtime/markitai/commit/2065df68e3b8b98631d13e66b9d8ab1c5f19d459))
* address third-party review of LLM warning implementation ([f390517](https://github.com/Ynewtime/markitai/commit/f3905177297ce6f2c7dd64dcea5af395b8ed26f2))
* align write_bytes_async fd handling with atomic_write_text_async ([bdf2fe2](https://github.com/Ynewtime/markitai/commit/bdf2fe226c995c20801576ff01895cb776859277))
* apply ruff format to cli/main.py and steam_news.py ([3ba6abb](https://github.com/Ynewtime/markitai/commit/3ba6abb8a23a6b2a360a9bc041c85c2b1e808d20))
* **auth:** platform-aware install hints and OAuth refresh token support ([8b7b4c3](https://github.com/Ynewtime/markitai/commit/8b7b4c3baecd48561d5df6488cf8a6cb001dcc33))
* batch mode shows skipped files in summary, SVG treated as image-only ([aa6fb6d](https://github.com/Ynewtime/markitai/commit/aa6fb6d1a85ece887a132a5229ac16c4e4f8f318))
* cache io_semaphore to prevent creating new instance per access ([a943fb4](https://github.com/Ynewtime/markitai/commit/a943fb45305d5925a67fe93c9ff2fc1be7c4ea2f))
* **cache:** close SQLite connections to prevent ResourceWarning on Python 3.13 ([965ccb5](https://github.com/Ynewtime/markitai/commit/965ccb59481eb606329e47d28171d4a1d9d77bc8))
* **cache:** prevent stale markitai_processed timestamp on cache hit ([a26b38c](https://github.com/Ynewtime/markitai/commit/a26b38c50030a30b633578d8f138758e6082b4a7))
* catch OSError from cairosvg import on Windows/macOS CI ([9f19499](https://github.com/Ynewtime/markitai/commit/9f19499a8cbced84a0afb7bd954c05df9eaa11c8))
* **ci:** add --all-groups to uv sync for explicit dev dependency install ([876935e](https://github.com/Ynewtime/markitai/commit/876935e934d70f8e8615971d5d272e5b7c52b8e1))
* **ci:** comprehensive test and publish workflow fixes ([61f6dc0](https://github.com/Ynewtime/markitai/commit/61f6dc0b118a2e89dd43e0f7910d661e65aca3bc))
* **ci:** revert deploy-pages to v4, v5 does not exist ([1d94d54](https://github.com/Ynewtime/markitai/commit/1d94d54844bed8bdef70b3a9828b8d5531097fe0))
* **cli:** add __main__.py for -m invocation, detect all providers in interactive ([ef8b915](https://github.com/Ynewtime/markitai/commit/ef8b9159dd1ee1566f35a406d69f7de276cfcad9))
* **cli:** allow custom preset names from config in --preset flag ([7f261c7](https://github.com/Ynewtime/markitai/commit/7f261c7d3e823a3d29cc8f4982a29c755f20389f))
* **cli:** cache __getattr__ result and remove empty TYPE_CHECKING blocks ([e437840](https://github.com/Ynewtime/markitai/commit/e4378406758b222d753ee8377f10de4bab720142))
* **cli:** format config get output as JSON with syntax highlighting ([48925b5](https://github.com/Ynewtime/markitai/commit/48925b51115abd8ceaaff9e1978a44108fe095a0))
* **cli:** improve doctor LLM API display and init provider detection ([244719f](https://github.com/Ynewtime/markitai/commit/244719fbf6ba02ec05ed97e35fa3653b3267058e))
* **cli:** improve version flag, JSON output, stale dates, and init display ([fa828f9](https://github.com/Ynewtime/markitai/commit/fa828f9b924b39bc014283978cdbaf61da5bcbf6))
* **cli:** improve Windows dependency detection and UI alignment ([0eadf7c](https://github.com/Ynewtime/markitai/commit/0eadf7cf386c8f3e6a605d07132ea1d0b9f62730))
* **cli:** revert version flag to -v/--version, keep --verbose without short flag ([2c3fb71](https://github.com/Ynewtime/markitai/commit/2c3fb71a5f160d551e93efd61586c85bbab2a588))
* **cli:** show warning when no LLM provider detected ([cc4f5b7](https://github.com/Ynewtime/markitai/commit/cc4f5b7fe0f88a7938b7b19b3b13e7151cde015b))
* **cli:** uncheck alt and ocr by default in interactive mode ([3f907e3](https://github.com/Ynewtime/markitai/commit/3f907e3416ad12690d66d93d1d72e93b75aa9364))
* **cli:** unify init and doctor terminal output style ([f0445c2](https://github.com/Ynewtime/markitai/commit/f0445c2cbd549f373d10580b6f272f3f266c0802))
* **cli:** use centralized console and fix URL path handling ([399c8aa](https://github.com/Ynewtime/markitai/commit/399c8aab9390bbb9859208aacba65dbd4706ef6e))
* **cli:** use sys.argv[0] for interactive mode subprocess ([86af521](https://github.com/Ynewtime/markitai/commit/86af521abe1bf4bb7941092746d99f9e13478cfb))
* complete deep audit — schema defaults, docs sync, docstring fix ([0df52b0](https://github.com/Ynewtime/markitai/commit/0df52b060494d0b567e78ed969a6b324e4f880c9))
* **config:** sync schema and function defaults with constants ([2ad7468](https://github.com/Ynewtime/markitai/commit/2ad74686f9e54808b8d2a3f2f2a784444dfc8912))
* convert images to PNG for Kitty graphics protocol ([4014187](https://github.com/Ynewtime/markitai/commit/4014187ef23de1c12bd3454e07574185a47544f0))
* copilot login, error message clarity, pyright warnings ([532034d](https://github.com/Ynewtime/markitai/commit/532034d308a7f4ab84c369dca88aba0348c201bd))
* **copilot:** add missing provider kwarg to ProviderError ([7ad1775](https://github.com/Ynewtime/markitai/commit/7ad17756f79399a20bea0422eac41b88796fd4e2))
* **copilot:** remove outdated GPT-5 series from unsupported models list ([c42e166](https://github.com/Ynewtime/markitai/commit/c42e166e0e9593d0f8e3a00490e2e884fc3ca2cc))
* correct content_profile enum in github_issue_thread fixture ([7eb81a2](https://github.com/Ynewtime/markitai/commit/7eb81a2f7aba892ae0a470dd1cbc4f6a2829be3a))
* cross-platform test failures for Windows and missing Playwright browsers ([4cb7118](https://github.com/Ynewtime/markitai/commit/4cb7118a7b7f6b6cc3575c430e9131fb16191094))
* deduplicate author links in X tweet extract_root path ([dbe4e49](https://github.com/Ynewtime/markitai/commit/dbe4e49112f1ba8f69cea51fcbd91808c432fb4c))
* deduplicate stabilization calls and add paged_stabilized guard ([4feac1c](https://github.com/Ynewtime/markitai/commit/4feac1cdd88e615847fbdad1f337fc0d9a631132))
* derive content_profile from matched extractor in generic path ([4d52274](https://github.com/Ynewtime/markitai/commit/4d522741ec9a6b3116d4ef6bba3b8a2b49a8ee05))
* detect CSS Modules hidden elements (e.g. isHidden-&lt;hash&gt;) ([487511b](https://github.com/Ynewtime/markitai/commit/487511b50b0a66dccdaaa136f6b4b8efbbe38dfa))
* disable pre-LLM deduplication by default to preserve original content ([c2f720d](https://github.com/Ynewtime/markitai/commit/c2f720d6fd20df7808bb6e29d09b5afd69fbdbf8))
* disable pymupdf4llm builtin OCR to suppress misleading Tesseract warnings ([057295a](https://github.com/Ynewtime/markitai/commit/057295a9c2b2a9be00eeba9f280d7935a4ba83a2))
* early return when tag.attrs is falsy (None or empty dict). ([3403105](https://github.com/Ynewtime/markitai/commit/34031058c84a01ae3e83ad13904b37dd15a05e17))
* **error:** preserve exception chains and log silent failures ([817bd10](https://github.com/Ynewtime/markitai/commit/817bd101c7f6020b2c69539cefb3d29b0c6907b7))
* **fetch_playwright:** force utf-8 for html markdown conversion ([a489e27](https://github.com/Ynewtime/markitai/commit/a489e274c165e5f03600e4078231cb1a7a092a0e))
* **fetch:** drop stale-loop HTTP clients at cleanup instead of awaiting aclose ([d6de457](https://github.com/Ynewtime/markitai/commit/d6de457d590f968117c20a7fb2a4582654ad170d))
* **fetch:** escape double quotes in DOM cleanup JS selectors ([3c66252](https://github.com/Ynewtime/markitai/commit/3c6625280f55b4937b376814666097fe24bc1f92))
* **fetch:** improve Playwright wait strategy and setup scripts ([438d2dd](https://github.com/Ynewtime/markitai/commit/438d2dd9b00cb62c05231de7342529631d9ab18d))
* **fetch:** never render blank error messages ([7a7970c](https://github.com/Ynewtime/markitai/commit/7a7970c8b69a6c7269cbf73f695c851d4415627d))
* **fetch:** rebuild HTTP clients when the event loop changes; harden CI-flaky tests ([9b31fbb](https://github.com/Ynewtime/markitai/commit/9b31fbb457b22fba4f157b5dad7018674a71fc15))
* **fetch:** switch Playwright default to domcontentloaded ([732ed34](https://github.com/Ynewtime/markitai/commit/732ed349a8f7b91e9176c7589284db9ec2431704))
* **fetch:** tweet URLs actually reach FxTwitter in the auto chain; rebuild X DOM extractor for the 2026 redesign ([114f3e5](https://github.com/Ynewtime/markitai/commit/114f3e5be83278508d620bac7da37cbec7171524))
* filter unreliable language field from external source metadata ([f7c8383](https://github.com/Ynewtime/markitai/commit/f7c83836ca7d53bfa961e66d87024a2144d5d11b))
* frontmatter regex, env quoting, Ctrl+C handling, hardcoded weight, docstring ([8cc61d4](https://github.com/Ynewtime/markitai/commit/8cc61d4f526d52be8e1e99a4a62f3d61582c8064))
* guard against decomposed elements in hidden element removal ([b3e53ba](https://github.com/Ynewtime/markitai/commit/b3e53badbefd01ca474800e9231e93df440c4e8c))
* handle None tag.attrs in webextract sanitize ([3403105](https://github.com/Ynewtime/markitai/commit/34031058c84a01ae3e83ad13904b37dd15a05e17))
* harden vision URL path with image placeholder fallback ([78f31f3](https://github.com/Ynewtime/markitai/commit/78f31f3505a35ad11988a1dcb69067bf86aeba76))
* image alt text language context, empty caption fallback, stabilize ordering ([1a38fdb](https://github.com/Ynewtime/markitai/commit/1a38fdb5a60477d8bc160cfa7f2a6b1023bf9907))
* **init:** improve setup UX, config path defaults, and load balancing ([ab5447a](https://github.com/Ynewtime/markitai/commit/ab5447a9a32e304bbb62c6c5c4260a777c408d88))
* interactive mode shows actual configured models instead of auto-detected provider ([d27df46](https://github.com/Ynewtime/markitai/commit/d27df46f46c6f6fe8583b302148159ef36acf100))
* **llm:** add JSON repair fallback for malformed instructor responses ([4dad140](https://github.com/Ynewtime/markitai/commit/4dad14073c8fa624e65141d14dcd6bbb2298c554))
* log debug message when resolver returns empty content ([10aa516](https://github.com/Ynewtime/markitai/commit/10aa516a618af6ab8b26f046fe2a3bba99f66d20))
* lower too_short threshold and add pure mode to interactive CLI ([81c10e8](https://github.com/Ynewtime/markitai/commit/81c10e84f82adfeb0ca69f690743951d5ef3319a))
* make LLM errors visible in quiet/stdout mode via ERROR-level handler ([2778037](https://github.com/Ynewtime/markitai/commit/277803784dc2b646abbc2eba85b96501f561fd1d))
* make LLM errors visible in quiet/stdout mode via ERROR-level handler ([1e32ed0](https://github.com/Ynewtime/markitai/commit/1e32ed0ab15c7340800f3bcf1e849086e56a069c))
* make write_bytes_async use atomic write pattern ([ce1a860](https://github.com/Ynewtime/markitai/commit/ce1a860a6cec66225b96ff5100102475df0130b9))
* migrate batch state files to .markitai/states namespace ([bb1987e](https://github.com/Ynewtime/markitai/commit/bb1987ee2e8396aeb9844806c963b0197cd2050b))
* **models:** handle C:/ forward-slash Windows paths in context_display_name ([8f1009d](https://github.com/Ynewtime/markitai/commit/8f1009d3fa4d0e09f374901dcc1a788e80a4aa0b))
* pin litellm to &lt;1.83.0 to avoid compromised versions ([16aca0e](https://github.com/Ynewtime/markitai/commit/16aca0e23ebb380cdb7073a5b4d4784640f029d5))
* pin litellm to &lt;1.83.0 to avoid compromised versions ([e419f4b](https://github.com/Ynewtime/markitai/commit/e419f4b9b08d6aa6309637735a398710cb739c68))
* **pipeline:** use fresh_soup_and_root for Level 1 extraction too ([ce1df89](https://github.com/Ynewtime/markitai/commit/ce1df89a897d65a9013ff37e2cff081189e6ee28))
* polish config editor nits and optimize init default config ([f5b2778](https://github.com/Ynewtime/markitai/commit/f5b2778d87b7134bdc6aa999580c81107dbdc034))
* prefer defuddle/jina over playwright for SPA domains ([8f5091c](https://github.com/Ynewtime/markitai/commit/8f5091c53e131269fc13656818e8619b54dc0edd))
* preserve image positions in standard LLM document flow ([3cfb993](https://github.com/Ynewtime/markitai/commit/3cfb993da7874453d02eca03b8181473011fc08b))
* prevent interactive OAuth during LLM calls — fail fast instead of blocking ([296963d](https://github.com/Ynewtime/markitai/commit/296963dfbd94990201568e779264f21e60c1298b))
* process_with_llm respects --pure mode (no LLM-generated frontmatter) ([12f88de](https://github.com/Ynewtime/markitai/commit/12f88defd371f5676e48862ef8074cee5d1be716))
* protect task lists and callouts from removal pipeline ([faae8b2](https://github.com/Ynewtime/markitai/commit/faae8b221d6fa8c2d52595932a2fa4e7dc138c39))
* provider auth preflight, install scripts extras parsing and resilience ([3aca307](https://github.com/Ynewtime/markitai/commit/3aca30707ae9f7cf3749b58a43417d4b6ee03c58))
* **providers:** align Claude Agent SDK usage with official docs, add env var auth detection ([a13d09a](https://github.com/Ynewtime/markitai/commit/a13d09a0be0a90a4e266226ceb051118cec3192c))
* pure mode bypass vision paths, warning false positive, type safety ([5c6f4ab](https://github.com/Ynewtime/markitai/commit/5c6f4ab0eb4ed6834f87ad89b637bee166b11880))
* pure mode reconstructs source frontmatter before LLM cleaning ([326e879](https://github.com/Ynewtime/markitai/commit/326e879b0ebaa529782e3b565e789cc95ca701c6))
* **removals:** protect structured content from trailing thin section removal ([4b80ce6](https://github.com/Ynewtime/markitai/commit/4b80ce629ac099903cf0b76b0462404865ddd283))
* replace regex with str.replace in PDF image path fixing ([a993311](https://github.com/Ynewtime/markitai/commit/a9933110947b430fbe6aef14629e4adaa78a7731))
* reset _detected_proxy_bypass in close_shared_clients, move import ([32395fd](https://github.com/Ynewtime/markitai/commit/32395fd9a9ceb7ac3a2cfaef9cc631d595f97f72))
* reset global semaphores and state in close_shared_clients() ([47b3de0](https://github.com/Ynewtime/markitai/commit/47b3de06c28e711f0311d65fbfe0f37b16cddc0c))
* residual placeholder cleanup was stripping image position placeholders ([f5f157a](https://github.com/Ynewtime/markitai/commit/f5f157a4ef2d5ddd277e48d70034e74a8d01c397))
* resolve all pyright errors and warnings in test files ([5720230](https://github.com/Ynewtime/markitai/commit/5720230f38b946532be0aa7a0e6b134c9d6bc4e7))
* resolve all Pyright warnings across codebase ([6486a7b](https://github.com/Ynewtime/markitai/commit/6486a7bc38215f63fb4d5356ed58e91f1a67dacd))
* resolve all Pyright warnings to achieve 0 errors 0 warnings ([2dcc09c](https://github.com/Ynewtime/markitai/commit/2dcc09c2f29a53b42320a8c5a405195f316c2232))
* resolve all ruff, pyright, pytest and bandit issues ([351f145](https://github.com/Ynewtime/markitai/commit/351f145241d497faec73961704c428392707b347))
* resolve CI failures — pyright guards and batch state test ([6d6d3cc](https://github.com/Ynewtime/markitai/commit/6d6d3ccffc4ad712a99d4e5dd66262e2c41c6ce8))
* resolve CI test failures across unit test suite ([43a0c69](https://github.com/Ynewtime/markitai/commit/43a0c69cccb8d8426651c0e252fdcab050021d81))
* resolve Gemini auth email via Google userinfo API fallback ([32a9204](https://github.com/Ynewtime/markitai/commit/32a9204353c4713194cc1863fc58d443a1605359))
* resolve multiple issues in setup scripts ([1c7d016](https://github.com/Ynewtime/markitai/commit/1c7d01656997c8a88c8cd6b0beee9400560af37b))
* resolve three bugs in stdout image handling ([bfaadee](https://github.com/Ynewtime/markitai/commit/bfaadee6d2c2e0fcf8a3c14c4c102f4fc1fae98f))
* restore deprecated compat shims and fix pyright test coverage ([9678bd6](https://github.com/Ynewtime/markitai/commit/9678bd6b00720587a907f44497348153aa6c8080))
* **scripts:** add clack-style summary to setup-dev.sh ([26105b5](https://github.com/Ynewtime/markitai/commit/26105b51f199b7cab2485ef35ea7539ac1f40eed))
* **scripts:** add LLM CLI pre-detection in shell scripts ([c20f68b](https://github.com/Ynewtime/markitai/commit/c20f68bcb5cde9bb3dfa8f2fe61a6f61142cd3fa))
* **scripts:** add Playwright pre-detection to dev shell scripts ([0a36778](https://github.com/Ynewtime/markitai/commit/0a36778c669e06ab1eefb78b5c6d38a2fad76766))
* **scripts:** fix print_summary formatting issues ([e13791f](https://github.com/Ynewtime/markitai/commit/e13791fad39c049cc575ee6458ba5ae37bba54fa))
* **scripts:** improve Playwright detection and simplify FFmpeg version output ([92205a5](https://github.com/Ynewtime/markitai/commit/92205a554db429a8286b75703e2cedb409ede563))
* **scripts:** improve user interaction in piped execution ([052cf98](https://github.com/Ynewtime/markitai/commit/052cf986a07b3ca80621f280c78d27eec5f96209))
* **scripts:** improve user interaction in piped execution ([ac11882](https://github.com/Ynewtime/markitai/commit/ac1188261208976bd329204207a67dca2d3c565e))
* **scripts:** remove unnecessary Node.js check from dev setup ([ee0d889](https://github.com/Ynewtime/markitai/commit/ee0d88974532a3d93cbed2b3539a9bd7e0bc732b))
* **scripts:** silence all package manager output in install scripts ([6373a7d](https://github.com/Ynewtime/markitai/commit/6373a7d266b1b85665c59f023a2e5aab8db8d5c0))
* **scripts:** silence verbose uv and pre-commit output in dev scripts ([ec9f45f](https://github.com/Ynewtime/markitai/commit/ec9f45f6427b580c99cae2fd85fe2a444dbf9c98))
* **scripts:** support worktree in dev mode detection ([2fbb441](https://github.com/Ynewtime/markitai/commit/2fbb441979ff423044ff8e14d76b63e9fb432278))
* **scripts:** unify ps1 output format with sh scripts ([651a64a](https://github.com/Ynewtime/markitai/commit/651a64a8a4bb4e91802812fe7449e617c83f31fb))
* **scripts:** unify user edition output format ([531a57e](https://github.com/Ynewtime/markitai/commit/531a57ed28705cf9378713d8c8128dc343d5b8a1))
* **scripts:** use %b format for color escape sequences in clack_note/clack_log ([a25094a](https://github.com/Ynewtime/markitai/commit/a25094a43245c89822853d9eb4e06c5f6f769ba2))
* **scripts:** use boolean returns in Install-UV functions ([ea63a0a](https://github.com/Ynewtime/markitai/commit/ea63a0adc23cf555c2605a75b6767d004ed33537))
* **scripts:** use markitai's uv tool environment for Playwright installation ([9d8bd6f](https://github.com/Ynewtime/markitai/commit/9d8bd6fa2275f36fedb38a01db801165e4b2498f))
* **security:** resolve macOS /tmp symlink false positive in check_symlink_safety ([722d863](https://github.com/Ynewtime/markitai/commit/722d8639145aa0306a4e2882fa9c78953c8f2d91))
* set USERPROFILE in test_expands_tilde for Windows compat ([4d43a03](https://github.com/Ynewtime/markitai/commit/4d43a03397a2653c83732f4ef6478b6caed16533))
* **setup:** accumulate extras in single uv tool install to prevent mutual overwrite ([30a553e](https://github.com/Ynewtime/markitai/commit/30a553eeee2a2df4b284c32a99fabbf0344ea5c8))
* **setup:** improve onboarding UX for .env config, interactive mode, and Windows execution policy ([2882452](https://github.com/Ynewtime/markitai/commit/288245242fb3a5be9b21d32cab21f485adf2d747))
* **setup:** use `uv tool upgrade` for existing installations ([66b9619](https://github.com/Ynewtime/markitai/commit/66b961958876dbbb78f75170189fd24943463d4a))
* show model info after LLM enablement prompt, not before ([71ffad1](https://github.com/Ynewtime/markitai/commit/71ffad150dad9efc4aa6c8269e49a812c9aab323))
* strip frontmatter before extracting image analysis document context ([c20c0e7](https://github.com/Ynewtime/markitai/commit/c20c0e79fa520db0a429b4c10d811c6b8efbfefe))
* strip orphan separator lines from markdown output ([cbbe309](https://github.com/Ynewtime/markitai/commit/cbbe309474ad7d2ec634d1f39b1396bced0032e8))
* sync config, CLI, and documentation with codebase state ([cb2e5b4](https://github.com/Ynewtime/markitai/commit/cb2e5b483a6e3f51ac358347712b51d3fd70eb03))
* **test:** add missing find_libreoffice mock in test_all_ok_message_when_complete ([fe235c4](https://github.com/Ynewtime/markitai/commit/fe235c469fe6da680ea9834084d6ec24eda7d124))
* **test:** filter known third-party pytest warnings (pydub ffmpeg, litellm async mock) ([360b3dc](https://github.com/Ynewtime/markitai/commit/360b3dc4d774cbdf346e96d677a9226093a1fa5e))
* **test:** remove unused variable in playwright test ([368f3d0](https://github.com/Ynewtime/markitai/commit/368f3d0e4eab9856ee3785ad2729b5719f456b2e))
* **tests:** CI-environment failures unmasked once unit flakes were cured ([f90f96d](https://github.com/Ynewtime/markitai/commit/f90f96d9621c0805579ad584b051a7f8f2034c64))
* **tests:** exclude .urls files from batch conversion fixture ([423a652](https://github.com/Ynewtime/markitai/commit/423a652f015117c3f9e3f41b0118dcf6020d0ae7))
* **test:** skip Esc binding test on Windows, fix Pyright null safety ([239448f](https://github.com/Ynewtime/markitai/commit/239448f15af72e3765469871eb6584c08f4cd7bd))
* **tests:** run the subprocess help test via python -m, not uv's exe shim ([8c00bd9](https://github.com/Ynewtime/markitai/commit/8c00bd9571e00dd7bcef8a922019b2c9d6eebd22))
* **tests:** scrub CI env for the subprocess help test on Windows ([a2a97bd](https://github.com/Ynewtime/markitai/commit/a2a97bd3227c97a6e197df6e865d646bc4b458dd))
* **test:** use actual fixture filenames in CF integration tests ([2bac339](https://github.com/Ynewtime/markitai/commit/2bac339926a84de09e5e26fb3d6107d19d10f5b9))
* tighten litellm pin to &lt;1.82.7 (1.82.7+ compromised) ([03b26a6](https://github.com/Ynewtime/markitai/commit/03b26a6a7c34374cd272625245eb88e8696601a1))
* tighten native webextract acceptance and resolver parity ([3bab3db](https://github.com/Ynewtime/markitai/commit/3bab3dbfbcf7ea0c9f195770ce1c72d9d6f45186))
* truncate titles longer than 120 chars at word boundary ([9631592](https://github.com/Ynewtime/markitai/commit/9631592df5ca5c24d0b07d32eadebb07c8d24851))
* **types:** wrap class_ lambda with bool() to satisfy pyright ([b5edac4](https://github.com/Ynewtime/markitai/commit/b5edac45417df9c8a7d853c5ea39b180074b725d))
* update /pyproject.toml ([4f1e353](https://github.com/Ynewtime/markitai/commit/4f1e3530f62952a99141f6a354060aa007e0396d))
* update schema and integration tests for output strategy optimization ([ec4b765](https://github.com/Ynewtime/markitai/commit/ec4b765f40552cc6c8223e4a7c4ac3d4afa22e37))
* URL mode without -o outputs to stdout instead of erroring ([993c37b](https://github.com/Ynewtime/markitai/commit/993c37bb121233253377ad500af4c1ab94adc261))
* URL processors respect --pure/--llm/--keep-base for base .md output ([c16e43a](https://github.com/Ynewtime/markitai/commit/c16e43a17fe312a4d57940ef81008297e2c280d5))
* use atomic write in ConfigManager.save() ([1b90422](https://github.com/Ynewtime/markitai/commit/1b90422a08372a57f4acbb80a911666114c78195))
* use cfg.cache.global_dir for all cache directories instead of hardcoded paths ([2e1ffcf](https://github.com/Ynewtime/markitai/commit/2e1ffcf5ecd53fee53e1f0cb520a3c2f0f401cbe))
* use original URL for extractor matching to survive redirects ([ee6c2d4](https://github.com/Ynewtime/markitai/commit/ee6c2d4e4aa63b1fcca7ed35ad6e330832e68f72))
* use Playwright in _fetch_multi_source() for screenshot mode ([308bd64](https://github.com/Ynewtime/markitai/commit/308bd64cc46268e2b21a5bfcc38740d1bd616bee))
* use tmp_path instead of /tmp in batch tests for macOS compat ([a92474c](https://github.com/Ynewtime/markitai/commit/a92474cf5db454356cf30a0671abaf75008ef871))
* warn user when LLM enabled but no provider found in interactive mode ([790317d](https://github.com/Ynewtime/markitai/commit/790317d02ffd5919dc1ecb4e338030130263cf11))
* **webextract:** address review findings — selector conflicts, math protection, identity checks ([80128bf](https://github.com/Ynewtime/markitai/commit/80128bfed4c2a1ade3627e53286d327829ab002b))
* **webextract:** callout syntax, task list checkboxes, layout table unwrapping ([6affd87](https://github.com/Ynewtime/markitai/commit/6affd87bd563a1856f1f5a601a996429859c4872))
* **webextract:** clean tweet-internal noise in XTweetExtractor ([8230203](https://github.com/Ynewtime/markitai/commit/8230203cdab4853430023be009bd08e3e2acd394))
* **webextract:** fix playwright crash on X.com, add tweet extractor ([4d1f828](https://github.com/Ynewtime/markitai/commit/4d1f8287f4112b9e9e18881f3f27259fd1d11c63))
* **website:** gitignore changelog.md build artifact, sync via docs:build ([b6f26ef](https://github.com/Ynewtime/markitai/commit/b6f26efb803241f1334f555e04b4cfb7345324a6))
* Windows compatibility, lazy dir creation, pyright warnings ([b888754](https://github.com/Ynewtime/markitai/commit/b888754c0360c1b3ba845451cce71273572b070f))
* wrong message index in vision json_mode and race condition in parallel gather ([4185c39](https://github.com/Ynewtime/markitai/commit/4185c3976b8822b32ca48a3600180fd3efa00af0))


### Performance Improvements

* bypass render_markdown in pipeline to eliminate extra BeautifulSoup parse ([4c5e525](https://github.com/Ynewtime/markitai/commit/4c5e5259a6f40068dc3590fded1125dc54ae647e))
* bypass render_markdown in pipeline to eliminate extra BeautifulSoup parse ([19c6190](https://github.com/Ynewtime/markitai/commit/19c61908eb54dd391f690266b29163f7b988ad0f))
* **cache:** merge stats and model breakdown into single query ([bd163d9](https://github.com/Ynewtime/markitai/commit/bd163d9c4cfbdb5a17dcd620bd76c7e628a86dbc))
* **cli:** avoid eager imports in --help by storing short help text ([4d24609](https://github.com/Ynewtime/markitai/commit/4d2460960072ee1c47655137b6cad30f0b0c8493))
* **cli:** lazy-load processor and command modules for faster startup ([b10fe2c](https://github.com/Ynewtime/markitai/commit/b10fe2c007855b813891745c122b908e757b7e49))
* **cli:** parallelize doctor and init dependency checks ([2977f9a](https://github.com/Ynewtime/markitai/commit/2977f9a147223680b83c4faed3e3f0465007b825))
* **executor:** auto-detect heavy task limit based on system RAM ([5f1de02](https://github.com/Ynewtime/markitai/commit/5f1de0225f082f14df4c26ea242630aebc4d4a38))
* **fetch:** add async-safe cache locking for concurrent URL fetching ([ac7d48d](https://github.com/Ynewtime/markitai/commit/ac7d48df34f5f3e5a551c1cdbefc8e42513ba855))
* incremental state saving via JSONL WAL pattern ([b4de183](https://github.com/Ynewtime/markitai/commit/b4de1831e40d5d21d1734c6002a28dbfdf2ff818))
* introduce ExtractionContext to eliminate redundant HTML parsing ([a7e75e4](https://github.com/Ynewtime/markitai/commit/a7e75e468cd8e075cb03f63995df26acac34fa50))
* introduce ExtractionContext to eliminate redundant HTML parsing ([23fa395](https://github.com/Ynewtime/markitai/commit/23fa395341c64a3052e53a1e981438096ecf70cc))
* join exact selectors into single CSS query ([0cc020c](https://github.com/Ynewtime/markitai/commit/0cc020c7af1aa8de98d2b2bafb459996154c0c0c))
* join exact selectors into single CSS query ([2466957](https://github.com/Ynewtime/markitai/commit/2466957e3abc3da28322dbe21ebe8f2df499e51a))
* lazy imports in cli, workflow, and processors __init__ to cut cold startup from ~5s to ~0.3s ([9bb1014](https://github.com/Ynewtime/markitai/commit/9bb10140b125117555604b72b498fb2462134ae3))
* **llm:** pre-compile regex patterns and batch replacements ([a915365](https://github.com/Ynewtime/markitai/commit/a9153650ff19e2e6568dcc2883ee780e950accda))
* **pdf:** extract parallel page rendering for standard and LLM modes ([c40206c](https://github.com/Ynewtime/markitai/commit/c40206cd01b87a3833a3eaf8506eb1a7596e5b08))
* **tests:** optimize test suite speed (~70s → ~30s) ([500c8d0](https://github.com/Ynewtime/markitai/commit/500c8d0c639f7d1d23db17e2f34e76ebfc70390d))
* **workflow:** offload CPU-intensive image processing to thread pool ([72622d2](https://github.com/Ynewtime/markitai/commit/72622d249a6467361abfb0b0b905a340e87fb96a))


### Reverts

* remove LLM-generated language field from Frontmatter ([f32a425](https://github.com/Ynewtime/markitai/commit/f32a4257e7593a6ed0a4d0d168134785da11cf05))

## [0.15.0] - 2026-07-04

Maintenance overhaul: full dependency refresh, Python 3.14 support, and a
multi-dimension audit that fixed 30+ verified bugs across batch processing,
fetch/cache, LLM providers, image handling, and configuration.

### Added

- **Python 3.14 Support**: `requires-python` relaxed to `<3.15`; full test suite passes on 3.14 (the previous onnxruntime blocker is resolved). CI matrix and classifiers updated
- **MIT License**: LICENSE file added and declared in package metadata (`License-Expression: MIT`)
- **Grouped `--help`**: options are organized into panels (Output & Configuration / LLM Enhancement / OCR / Fetch & Conversion Backends / Batch Processing / Cache & Images / Logging & Info) via rich-click; lazy subcommand loading preserved so `--help` stays ~100ms
- **Garbled-text detection**: PDFs whose extracted text is unreadable (broken cmap/substitution ciphers) are detected via a CJK-safe vowel-ratio heuristic
- **Scan/garbled advisory**: converting a PDF with scanned-looking or garbled pages without `--ocr` now emits one consolidated warning naming the affected pages and suggesting `--ocr`
- **Repeated header/footer suppression**: running headers/footers (incl. "Page N of M" patterns) repeated across ≥60% of pages are stripped from PDF output — cleaner Markdown, fewer wasted LLM tokens; headings and tables are never touched, <4-page documents exempt
- **VLM degeneration guard**: vision/screenshot extraction results are checked for repetition loops (a known VLM-OCR failure mode); degenerate tails are truncated with a warning and never persisted to cache, so retries aren't poisoned
- **HTML extraction quality (ported from Defuddle upstream)**: MathJax `script[type="math/tex"]` equations preserved as LaTeX; Wikipedia/MediaWiki MathML survives hidden-element removal; partial-selector clutter removal no longer deletes code blocks (`<pre>`-protection); anchor-wrapped headings unwrap cleanly; code-block language tags validated against an allowlist (no more ```codeblock); whitespace inside `<pre>` preserved
- **Footnote engine (full Defuddle port)**: footnotes/citations across Wikipedia, arXiv, Substack, WordPress, Word/Google Docs exports, Tufte sidenotes and more are standardized and emitted as real Markdown footnotes (`[^1]` / `[^1]: ...`) with renumbering, duplicate-reference handling, multi-paragraph definitions, and back-link stripping — 15 ground-truth fixtures now match Defuddle's expected output
- **Unified fetch strategy flag**: new `-s/--strategy auto|static|playwright|defuddle|jina|cloudflare`; the five per-backend flags (`--playwright`, `--defuddle`, `--static`, `--jina`, `--cloudflare`) remain as deprecated aliases that print a migration notice
- **Remote-fetch consent**: URLs are no longer sent to third-party extraction services (defuddle.md, Jina, Cloudflare) without consent — `fetch.remote_consent: ask|always|never` (default `ask`: interactive runs prompt once per process; non-interactive/quiet runs skip remote and crawl locally); `MARKITAI_NO_REMOTE_FETCH=1` forces `never`; explicit `-s defuddle`/`-s jina` counts as consent
- **PDF hidden-text sanitization**: invisible text (white-on-white, <2pt, zero-opacity, off-page) — a prompt-injection vector for LLM pipelines — is detected; `security.pdf_sanitize: off|warn|remove` (default `warn` logs a consolidated advisory naming pages)
- **Per-page OCR routing**: `--ocr` on mixed digital/scanned documents keeps the native text layer for digital pages and only OCRs scanned/garbled ones (`ocr.per_page_routing`, default on)
- **Conversion-quality benchmark harness**: dev-only `packages/markitai/benchmarks/` scores the HTML pipeline against the Defuddle ground-truth corpus (rapidfuzz block alignment + order score, marker-style); committed baseline: mean 91.04 over 83 fixtures
- **Release automation**: release-please drives versioning/changelog from conventional commits (release = merge the release PR; publishing chains via workflow dispatch); PR coverage comments via py-cov-action, no external service. Operational note: tag the 0.15.0 release manually first (or set `bootstrap-sha`) so release-please anchors correctly

#### Hunt round 5 (tweet pipeline root fix & flow polish)

- **FxTwitter now actually serves default tweet fetches**: the intercept only existed in the top-level PLAYWRIGHT dispatch branch — the default auto chain reached playwright through its own loop and skipped FxTwitter entirely, dropping users onto noisy DOM extraction. Fixed in the chain (with telemetry/return contract); regression tests pin the intercept
- **X DOM extractor rebuilt for X's 2026 redesign**: X removed every `data-testid` and moved to React+Tailwind markup, so the semantic tweet extractor never matched and the generic pipeline leaked avatars, cookie banners, stats bars, and blob: video links. The extractor now handles both the new (`data-tweet-id`, hover-card slots, permalink-text timestamps) and legacy markup, verified against a committed real-DOM fixture and live; the x.com domain profile's wait selector no longer burns a 10s timeout per tweet
- **"All fetch strategies failed" is never empty**: five silent skip paths (JS-detected static, consent-gated remote, missing playwright/browser, missing CF credentials) now record their reason; an all-skipped chain explains itself
- **`markitai init` merges instead of dead-ending**: with an existing config the wizard offers Update (default) / Overwrite / Keep — Update non-destructively appends newly detected providers; `init -y` applies it automatically and reports what changed. Auth login hints are config-aware ("adds it to your existing config" / "Already enabled in `<path>`"). Dependency + provider detection now run concurrently under a spinner (Gemini's userinfo call, up to 5s, no longer serializes the flow)
- **Login failure cards**: provider login failures render the status-card style with context-aware hints (never suggesting the command that just failed; install command first); gemini login no longer dumps a raw traceback
- **`mkai` PyPI alias stub dropped**: PyPI's name-similarity guard rejects `mkai` (existing `mk-ai` project) — the same guard blocks would-be squatters, which was the stub's purpose; the `mkai` command itself still ships with markitai

#### Hunt round 4 (UX & quality polish)

- **Tweet conversion at defuddle parity**: the FxTwitter path (which serves default x.com fetches, with playwright as fallback) and the DOM extractor were both reworked — bold `**Name @handle** · date` author line, paragraphs preserved, t.co links expanded, video rendered as poster + link (was a broken mp4 embed), quoted tweets as blockquotes with author/date/media/permalink, author threads joined into the post body, card previews. Corpus mean 91.04 → 92.26
- **Live progress feedback**: long single-input conversions show a pure-ASCII stage-aware spinner on stderr (`Fetching (static)…` → `Rendering (playwright)…` → `Enhancing with LLM…`, bridged from fetch-stage logs); suppressed for pipes/--quiet/-v; stdout stays pure. Root cause of the "looks stuck" complaint: the old spinner machinery was constructed disabled in file-output mode
- **`markitai auth` status cards**: all four providers render a unified glyph card (✓/✗ login state, CLI/SDK presence, usage + next-step); bare `markitai auth` shows an all-provider overview; ChatGPT guidance now points at `markitai auth chatgpt login` (device-code flow verified live) instead of "pip install litellm"
- **Fetch errors are never blank**: exceptions with empty messages (e.g. httpx.ConnectError) now render their type via format_error_message across all fetch strategies

#### Hunt round 3 (release prep)

- **`.eml` email support (native, zero deps)**: headers/body/attachments via stdlib `email`; HTML bodies go through the standard HTML pipeline, image attachments flow into the assets/vision pipeline, nested messages render quoted (depth 1); header values sanitized against injection. EML no longer delegates to kreuzberg
- **HEIC/HEIF/AVIF input**: new `markitai[heif]` extra (pillow-heif); 12-byte ftyp sniff, decode-to-PNG at the boundary with EXIF orientation applied, then the normal OCR/vision/compression pipeline — iPhone photos just work
- **Quality guardrails gate**: `benchmarks/guardrails.json` pins a per-fixture minimum score (0.9 × current) plus corpus/local mean floors; `--check` fails CI (new ~1min job) when extraction quality regresses; `--update-guardrails` regenerates deliberately
- **`--config-json '<json>'`**: inline config overrides for agents/CI — merged over the config file, under explicit CLI flags
- **Subcommand help polish**: all 26 subcommand helps now render rich panels with Examples; empty states got helpful hints (`cache stats` with no cache, silent `init -y`, `config get` unknown key → list hint); action commands print the natural next step
- **`mkai` short command**: ships with markitai as a second console script. (A separate PyPI alias package was evaluated and dropped: PyPI's name-similarity guard rejects `mkai` because `mk-ai` exists — the same guard equally prevents anyone else from squatting the name)
- **kreuzberg floor raised to >=4.9.6**: picks up the no-OCR-backend PDF fix (4.7.3), image-heavy-PDF hang fixes (4.9.x), PPTX slide-order fix (4.8.0). Note: kreuzberg >=4.8.0 is Elastic License 2.0 (optional extra; accepted)
- **Cache-hit visibility**: the `Fetched via <strategy>` line notes `(cached)` — a cached defuddle result had masqueraded as the live default strategy (fresh fetches win with `static`)
- Dependency patches: litellm 1.90.3, rapidocr 3.9.1

#### Hunt round 2 (follow-up fixes)

- **Playwright browser detection fixed**: newer Playwright ships `Google Chrome for Testing.app` instead of `Chromium.app`, so the path-only check reported "browser not found" even after a successful install; detection now uses Playwright's own `INSTALLATION_COMPLETE` marker (bundle-name/version agnostic) with executable paths as fallback. The `uv tool run --from 'markitai[all]'` install hint was dropped (triggers a uv warning and resolves an ephemeral env whose Playwright version can mismatch)
- **Remote-fetch consent is now lazy**: the consent prompt fired before the chain ran, so even URLs satisfied by the local-first chain asked "Allow sending URLs to remote services?"; consent is now resolved only when a remote strategy is actually about to run — local successes never prompt
- **Config filename unified in all messages**: hints/errors that said "in markitai.json" (doctor's LLM hint, the Cloudflare workflow error) now show the actually-loaded config path, matching the doctor header
- **`-h` paragraph spacing normalized**: rich-click renders `\b` blocks with two trailing blank lines but plain paragraphs with none; docstring now renders with exactly one blank line between all sections (`TEXT_PARAGRAPH_LINEBREAKS`)
- **Dependency patch bumps**: litellm 1.90.3, rapidocr 3.9.1

#### CLI & fetch polish (post-release hunt round)

- **`mkai` short alias**: installed alongside `markitai` (verified conflict-free on PyPI/homebrew/system)
- **`-b/--backend native|kreuzberg|cloudflare`**: file conversion backend is now its own orthogonal flag; `--kreuzberg` remains as a deprecated alias. `-s` is purely the URL fetch strategy
- **Local-first auto chain**: default order is now `static → playwright → defuddle → jina → cloudflare` (static's native extraction matches remote defuddle on the ground-truth corpus and beats it on CJK spacing); SPA/JS-heavy domains go straight to the browser
- **`markitai doctor` first run 36s → ~5s, warm 1.0s → 0.33s**: two root causes — (a) the RapidOCR check imported the real module, pulling in opencv's 119MB dylib whose one-time macOS dyld signature validation cost ~25s on a fresh install (now probed via package metadata only, no import); (b) an unconditional litellm import cost 0.55s even with no models configured (now deferred). Output normalized (consistent inline item format, single-blank-line sections, failure summary uses ✗ not ✓) and now shows the loaded config file path
- **Actionable configuration errors**: Cloudflare credentials, Playwright missing-browser, and Jina auth errors now include the concrete config file path, copy-pasteable `markitai config set`/env commands, and credential acquisition steps (token URL + required permissions)
- **Jina refusal fallback**: service refusals (e.g. github.com 451 anonymous block) no longer dump raw JSON — interactive runs are asked once per run whether to fall back to the auto chain (default yes); non-interactive runs fall back automatically with a warning
- **Claude subscription detection fixed on macOS**: Claude Code stores OAuth tokens in the Keychain, not `~/.claude/.credentials.json` — markitai reported "not authenticated" on logged-in machines and `init` silently skipped claude-agent. `claude-agent/` models use the Claude subscription quota via the local CLI (no API key needed); `markitai auth claude status` now shows identity, plan, CLI/SDK state and a config snippet
- **stdout no longer hard-wraps content**: Rich's console.print wrapped output at terminal width, breaking long URLs mid-token in piped output; content is now written raw
- **Help panels aligned**: metavar column removed (appended to help text instead); deprecated aliases use terse uniform descriptions
- **HTML extraction**: GitHub repo pages now extract just the README (was: file-tree tables, About sidebar, star counts — ~950 junk words); frontmatter gains `published` date, full untruncated titles without site suffixes, and no longer emits homepage canonical_url on article pages; bilibili/Twitter-widget iframes survive as links (root cause: embed canonicalization ran after sanitize stripped iframes). Benchmark mean 91.86 → 92.24, embedded-videos fixture +31 points, zero regressions; local self-baseline fixtures (GitHub repo + CJK blog) added to the benchmark
- **Xberg (kreuzberg successor)**: evaluated — PyPI `xberg` is currently a placeholder aliasing kreuzberg and real 1.0 wheels aren't published; kreuzberg v4 stays (LTS, maintained), with a documented migration checklist in the converter for when Xberg 1.0 ships

### Changed (behavior)

- **Mixing INPUT with a subcommand is now an error**: `markitai note.txt config list` previously dropped note.txt silently; it now fails with guidance. A file named like a subcommand (`config`, `doctor`, ...) gets a stderr hint to use `./config`
- **`-o out.md` with a single file/URL writes that file**: previously it silently created a directory named `out.md`; batch/directory input with a file-looking `-o` now errors clearly
- **Diagnostics go to stderr**: warnings/notices no longer pollute piped stdout output (`markitai x --alt | pandoc` receives pure markdown)
- **Output naming replaces the extension**: `sample.pdf` → `sample.md` (was `sample.pdf.md`). Colliding batch inputs (`a.pdf` + `a.docx`) and outputs that would overwrite the source keep the legacy `<name>.<ext>.md` scheme, per file. Re-running an old batch re-converts once under the new names
- **stdout image links now survive the process**: `image.stdout_persist` defaults to on (assets persisted under `~/.markitai/assets`); absolute temp-dir links from the PDF pipeline are normalized; opting out prints an ephemeral-links warning to stderr
- **Reports are batch-only by default**: single-file/single-URL conversions no longer write `.markitai/reports/` unless `output.report = true` (tri-state; `false` disables even for batches)
- **Repo hygiene**: AI-session artifacts (`.claude/` memory, `docs/superpowers/` working plans containing local dev paths) removed from the repository and gitignored

### Fixed

#### Batch & Workflow (correctness of success/failure reporting)

- **LLM failures no longer masquerade as success**: LLM API failures previously returned success-shaped results — batch marked files COMPLETED with `cache_hit=true` pointing at `.llm.md` files that were never written, and `--resume` skipped them. Failures now propagate, the file is marked FAILED, and the base-markdown fallback path actually runs
- **Resume re-processes interrupted files**: files that were IN_PROGRESS at crash time were silently dropped on `--resume` (never re-queued, counted in no summary bucket); they are now converted to FAILED on state load and re-processed
- **Per-file LLM cost attribution**: usage contexts are cleared after each file, fixing double-counted costs for same-basename files in different directories
- **Base64 image index desync**: an undecodable data URI shifted every subsequent image reference one position, attaching wrong images to wrong locations; extraction and replacement now apply the same skip rule
- **Path traversal via custom output names**: a `.urls` entry with a crafted output name (`../../x` or absolute path) could write converted output outside the output directory; custom names are now sanitized like auto-derived ones

#### Fetch & Cache

- **AUTO-strategy cache revalidation**: HTTP validators (ETag/Last-Modified) were discarded on the default fetch path, so cached pages were served stale forever; validators are now stored and conditional revalidation works as designed
- **Playwright context leaks**: cookie-validation errors leaked browser contexts; a `new_page()` failure in persistent mode raised `UnboundLocalError` masking the real error; concurrent same-domain fetches could double-create or double-close cached contexts (now lock-protected)
- **HTTP client cleanup**: old clients are no longer closed via unreferenced fire-and-forget tasks that asyncio could garbage-collect before running
- **URL list robustness**: a `null` or non-string `url` in a JSON URL list crashed the whole batch; it is now skipped with a warning
- **Proxy auto-detection**: SOCKS-only ports (Tor 9050, SOCKS5 1080, V2Ray 10808) were mislabeled as HTTP proxies and are removed from detection; detected proxies now log at WARNING

#### LLM & Providers

- **Batch vision deadlock**: language rewrites re-acquired the already-held concurrency semaphore — with `llm.concurrency=1` a single rewrite hung the whole pipeline; rewrites now run after the semaphore is released
- **Vision cache poisoning**: batch results were zipped positionally with requested images; a skipped/reordered model response persisted the wrong analysis under the wrong image's content hash across sessions. Results are now aligned by the echoed `image_index`, and ambiguous batches skip cache persistence
- **Copilot concurrent temp-file races**: the singleton provider's shared temp-file list let one request's cleanup delete another in-flight request's image attachments; tracking is now per-request
- **gemini-cli rate-limit failover**: a 429 raised a non-retryable error that aborted the request instead of retrying on another pool model; it now raises litellm's retryable `RateLimitError` (cooldown still recorded)
- **Event-loop stalls**: blocking token refreshes (ChatGPT/gemini-cli auth) and CPU-heavy native HTML extraction now run in worker threads instead of freezing all concurrent tasks
- **Retry backoff released**: exponential-backoff sleeps no longer hold a concurrency-semaphore slot, so a rate-limit burst can't collapse throughput of healthy models
- **Dynamic max_tokens**: the retry path now takes the minimum output limit across the model pool (matching instructor call sites) instead of the top-weight model only
- **Screenshot extraction cache**: keyed by content fingerprint instead of filename, so re-fetches of changed pages aren't served stale extractions

#### Images & Conversion

- **EXIF orientation**: rotation is baked in before re-encoding on all compression paths (OpenCV and Pillow); phone photos no longer come out sideways
- **LA-mode transparency**: grayscale+alpha images composite onto white when converting to JPEG instead of dropping the alpha channel
- **Uncompressed image naming**: with `image.compress = false`, original bytes are now saved under their actual format's extension/MIME instead of the configured output format's
- **EMF/WMF conversion failures**: unconverted EMF bytes are no longer mislabeled as PNG; failures now log a visible warning
- **OCR engine consistency**: a failed engine rebuild no longer leaves the old engine permanently served under the new config's fingerprint
- **Temp directory leaks**: converter paths that render page/slide images without an output directory now clean up their temp directories at process exit

#### Configuration & CLI

- **Config editor validation**: `markitai config edit` validated nothing before saving — an out-of-range value bricked every subsequent CLI invocation (including the editor itself). The editor now validates before save, and config loading reports a clear actionable error instead of a raw traceback
- **Symlink safety check**: the nested-symlink branch inspected the resolved path (which by definition has no symlinks) and never fired; it now walks the original path's ancestors
- **Config bounds**: `llm.concurrency` requires `>=1` (a persisted `0` hung every LLM task forever); router retry/timeout fields gained sane lower bounds
- **`config set` type coercion**: values are coerced by the target field's declared type — string fields keep leading zeros (API keys), bool fields accept `1`/`0`
- **`config set` bracket notation**: `llm.model_list[0].litellm_params.weight` now works for set (previously only get)
- **JSON log format**: log lines are built with proper JSON serialization; messages containing quotes/newlines no longer produce invalid JSON
- **`cache clear` prompt**: shows the actual configured cache directory instead of a hardcoded path
- **`config get` null handling**: existing-but-null fields print `null` and exit 0 instead of "Key not found" exit 1
- **Missing config path visibility**: a nonexistent `MARKITAI_CONFIG`/explicit config path now warns (and `~` is expanded) instead of silently running with defaults
- **Loguru misuse**: printf-style logging calls that silently dropped the URL and traceback now use loguru idioms

#### Post-review hardening

- **Parallel LLM task isolation**: a document-processing failure no longer leaves the sibling image-analysis task running detached (and vice versa); image-analysis failure now degrades gracefully (`.llm.md` kept without alt text) instead of failing the whole file
- **Usage cleanup on vision fallback paths**: partial LLM usage is cleared when vision enhancement fails, so it isn't attributed to the next file
- **`markitai doctor` exit code**: exits 1 when required dependencies are missing (previously always 0, and the summary claimed success); failure summary now reports the missing count
- **Symlink check refinement**: root-owned symlinks on POSIX (e.g. `/var/run -> /run`) are treated as OS artifacts instead of raising false positives

### Changed

- **Dependency refresh**: all dependencies upgraded ~4 months forward, including litellm 1.82.6 → 1.90.x, opencv-python 4.x → 5.x, starlette → 1.x, claude-agent-sdk 0.1 → 0.2, github-copilot-sdk 0.2 → 1.x, instructor 1.15, playwright 1.61, pymupdf 1.28. Test suite fully green on the new set. (markitdown stays at 0.1.5 — 0.1.6 requires a pre-release azure dependency; rich stays at 14.x — capped by instructor `<15`)
- **Version single-sourcing**: the package version is now read from `src/markitai/__init__.py` at build time (`dynamic = ["version"]`); no more triple-bump
- **Release guard**: `publish.yml` verifies the release tag matches the built version before publishing
- **Dependabot on uv ecosystem**: lockfile-aware dependency PRs (previously the pip ecosystem produced PRs that always failed `uv sync --frozen`)
- **README**: rewritten with install instructions (uv tool/pipx), extras table, and quick start — this is also the PyPI long description
- **CONTRIBUTING.md**: new contributor guide (dev setup, commands, conventions, release steps)
- **pre-commit**: pyright moved from per-commit to pre-push (full-project check was tens of seconds per commit)
- **`.env.example`**: bilingual (EN/zh) comments
- **bs4 4.15 compatibility**: attrs-only `find`/`find_all` calls pass an explicit tag matcher; `NavigableString` imported from `bs4.element` (upstream `__all__` regression)
- **Ruff target aligned to floor**: `target-version = "py311"` (was py313, which could suggest syntax breaking 3.11 support)
- **Modernized asyncio idioms**: `asyncio.get_event_loop()` → `asyncio.get_running_loop()` in async code

### Security

- **litellm supply-chain pin lifted**: `litellm>=1.83.0` replaces the `<1.82.7` emergency pin — the March 2026 compromise affected only 1.82.7/1.82.8, upstream audited 1.78.0–1.82.6 clean, and releases are signed since 1.83.0

## [0.14.0] - 2026-03-25

### Added

- **Steam News Extractor**: Site-specific extractor for `store.steampowered.com/news/` pages that parses BBCode announcements from JSON data attributes
- **MathML-to-LaTeX Converter**: Structural MathML conversion for pages without LaTeX annotations (KaTeX/MathJax), handling `msup`, `msub`, `mfrac`, `msqrt`, `mover`, `munder`, `mtable`, and 70+ Unicode math symbol replacements
- **LibreOffice Functional Check**: `is_libreoffice_functional()` verifies LibreOffice can actually convert files, not just that the binary exists
- **CSS Modules Hidden Detection**: Detect hashed hidden class names like `isHidden-vzcyV0` from CSS-in-JS frameworks

### Fixed

- **Math Content Extraction**: Body fallback now triggers when all retry levels fail to reach the sparse threshold, fixing KaTeX pages where scoring selected a single math div instead of the full article
- **Integration Test Reliability**: Batch test fixture filters to files with registered converters; LibreOffice tests skip properly when installation is non-functional
- **CLI Preset Validation**: Unknown presets now show available options and exit with error instead of silently continuing
- **BBCode XSS Prevention**: Raw HTML in Steam BBCode content is escaped before conversion to prevent injection

### Security

- **litellm Supply-Chain Pin**: Pin litellm to `<1.82.7` to exclude compromised versions

### Changed

- **CI Resilience**: Windows LibreOffice install retries up to 3 times with backoff to handle transient Chocolatey failures

## [0.13.1] - 2026-03-23

### Added

- **Config Editor Redesign**: Replace questionary select with a custom prompt_toolkit UI featuring a visible search box with frame, fuzzy filtering, scrollable list with cursor, and "↑ N more above / ↓ N more below" scroll indicators
- **Fuzzy Match Search**: Case-insensitive fuzzy matching for config settings (characters in order, not necessarily consecutive) with scoring that rewards consecutive and early matches
- **Config Field Descriptions**: Add `Field(description=...)` to 66 Pydantic config fields, displayed inline in the config editor
- **In-Place UI Refresh**: Use ANSI cursor position queries to erase only the lines occupied by each UI component, preserving terminal history

### Fixed

- **Esc Key Support**: Inject Esc key bindings into all questionary prompts (text, select) via prompt_toolkit `merge_key_bindings`; questionary 2.1.1 `select()` only binds Ctrl+C/Ctrl+Q natively
- **Bool Editor**: Replace `questionary.confirm()` with `questionary.select()` using `Choice(value=True/False)` for consistent Esc support
- **Search + j/k Conflict**: Disable `use_jk_keys` when `use_search_filter` is enabled (questionary 2.1.1 raises `ValueError` otherwise)
- **Literal Type Preservation**: Use `Choice(value=original)` to preserve original typed values (int, str) when editing Literal fields, instead of converting to string

## [0.12.1] - 2026-03-22

### Added

- **Stdout Terminal Image Display**: Inline image rendering for Kitty/iTerm2 terminals in stdout mode, with three-tier resolution cascade (terminal protocol → persistent asset store → markdown placeholder)
- **Content-Addressed Asset Store**: Persistent image storage with symlink refs at `~/.markitai/assets/`, enabling stdout image persistence across sessions
- **Terminal Image Protocol Detection**: Auto-detect Kitty and iTerm2 graphics protocols for native inline image display
- **`stdout_persist` Config Fields**: New `image.stdout_persist` and `image.stdout_persist_dir` settings for controlling stdout image persistence
- **External Image Inline Display**: Download and inline-display external images in single URL stdout mode (`image.stdout_fetch_external`)
- **User Journey Documentation**: Comprehensive Chinese user journey document covering all features and workflows

### Fixed

- **Stdout Mode LLM Errors**: Make LLM errors visible in quiet/stdout mode via ERROR-level log handler
- **LLM Warning Implementation**: Address third-party review findings on LLM warning display
- **Kitty Graphics Protocol**: Convert images to PNG for Kitty protocol compatibility
- **Stdout Image Handling**: Resolve three bugs in stdout image asset resolution and display
- **Cross-Platform Tests**: Fix Windows test failures and missing Playwright browser handling
- **`markitai init` Duplicate Routes**: Deduplicate overlapping default provider entries in generated configs, preferring Claude CLI over Anthropic API and direct Gemini API over OpenRouter Gemini

### Changed

- **Stdout Asset Resolution**: Rename `strip_asset_references` to `resolve_asset_references` with three-tier cascade logic
- **Terminal Image Rendering**: Harden rendering pipeline and improve test coverage
- **`markitai init` Default Config**: Stop writing redundant default `image.compress` and `image.quality` settings into newly generated configs

## [0.12.0] - 2026-03-20

### Added

- **Native HTML Extraction Parity**: Introduce resolver-based extraction pipeline with typed extraction results, frontmatter builder, quality profiles, and semantic models for threaded pages
- **Structured Site Extractors**: Rebuild threaded extraction on shared abstractions and add native resolver coverage for GitHub Discussions, X threads, and YouTube pages
- **Webextract Quality Enhancements**: Add noise removal, enhanced scoring, standardization, multi-level retry, content patterns, heading anchors, callouts, srcset optimization, and code language detection
- **CLI Force Flags**: Add `--static` to force static HTTP with native webextract and `--kreuzberg` to force kreuzberg conversion for all formats
- **Async Enrichment Pipeline**: Add policy-aware enrichers and thread inclusion rules for structured extraction
- **Language-Aware Vision Retry**: Retry and rewrite image analysis outputs in the document language

### Fixed

- **URL Stdout Fallback**: URL mode without `-o` now writes to stdout instead of erroring
- **Concurrency Safety**: Make `ContentCache`, `_image_cache`, model cooldown tracking, and `io_semaphore` thread-safe and reuse the cached semaphore instance
- **Atomic Writes**: Use atomic write patterns for `ConfigManager.save()` and async byte writes
- **Resource Cleanup**: Reset semaphores and proxy-bypass state in shared-client cleanup
- **Observability**: Add debug logging for previously silent exception handlers
- **Webextract Regressions**: Fix `None` `tag.attrs`, selector conflicts, math protection, callout/task-list/table formatting, X.com Playwright crash, tweet noise, and resolver acceptance parity
- **Tooling Hygiene**: Resolve remaining Ruff, Pyright, Pytest, and Bandit issues and close low-priority parity coverage gaps

### Changed

- **HTML Conversion Path**: Route HTML files through the native webextract pipeline by default
- **Fetch Internals**: Split `fetch.py` into smaller modules and decompose `fetch_url()` into composable sub-functions
- **CLI Logging UX**: Improve batch progress reporting and quiet/verbose URL logs
- **Release Cleanup**: Update dependencies, CI and website docs, model metadata, and clean up project structure for the `0.12.0` release

### Removed

- **Obsolete Project Docs**: Remove outdated root docs, archived plans, and historical reference material during project cleanup

## [0.11.2] - 2026-03-14

### Fixed

- **Windows Compatibility**: Add Windows `GlobalMemoryStatusEx` RAM detection for proper heavy task semaphore sizing
- **Lazy Directory Creation**: Defer `~/.markitai/` directory creation from import-time to first write — prevents side effects when the tool is only imported or used read-only
  - `SPADomainCache`: mkdir moved from `__init__` to `_save()`
  - `SQLiteCache`: mkdir moved from `__init__` to `_get_connection()` with `_dir_ensured` flag to avoid repeated syscalls
- **Default Output/Log Dir**: `DEFAULT_OUTPUT_DIR` and `DEFAULT_LOG_DIR` now default to `None` instead of hardcoded paths — output directory must be explicitly specified via CLI `-o` or config file
- **Pyright Warnings**: Eliminate all 27 pyright warnings — suppress `reportUnsupportedDunderAll` for PEP 562 lazy-loading modules, fix `curl_cffi` `ProxySpec` TypedDict type mismatch
- **Schema Sync**: Update `config.schema.json` to match new `OutputConfig.dir` and `LogConfig.dir` nullable types

## [0.11.1] - 2026-03-14

### Added

- **Interactive Pure Mode**: Add pure mode option to interactive CLI wizard

### Fixed

- **Pure Mode Vision Bypass**: `--pure` now correctly skips screenshot-only and vision enhancement paths, falling through to text-only LLM processing
- **Pure Mode Warning False Positive**: `--pure --screenshot-only` no longer warns about `--screenshot` being ignored
- **URL Content Validation**: Lower `too_short` threshold from 100 to 30 characters — minimal landing pages were incorrectly rejected after stripping markdown syntax
- **Type Safety**: Fix `merge_llm_usage` parameter type to accept `LLMUsageByModel` (pyright warning)
- **Dead Code**: Remove unused `_format_standalone_image_markdown` alias

### Changed

- **CI**: Upgrade GitHub Actions to Node.js 24 compatible versions

## [0.11.0] - 2026-03-13

### Added

- **Pure Mode (`--pure`)**: Full implementation of transparent LLM pass-through mode — text cleaning only, no frontmatter generation or post-processing
- **Pure Mode Decoupled from LLM**: `--pure` no longer implies `--llm`; `--pure` alone writes raw markdown without frontmatter, `--pure --llm` sends content through LLM cleaning only
- **Image Vision in Pure Mode**: `--llm --pure` with image inputs routes to Vision analysis path (`process_image_with_vision_pure`)
- **`--keep-base` CLI Option**: Explicitly write base `.md` even in LLM mode (default: skip base `.md` when LLM is enabled)
- **Image-Only Format Handling**: Skip image-only formats (PNG, JPG, etc.) in non-LLM/non-OCR mode with clear warning
- **LLM Fallback**: Write `.md` as fallback when LLM processing fails
- **Batch Skip Summary**: Group skipped files by reason with example filenames in batch summary
- **Pure Mode Warning**: Warn when `--pure` silently overrides `--alt`/`--desc`/`--screenshot`
- **Mode-Specific Cleaner Prompt**: `{mode_rules}` template variable in cleaner prompt — standard mode gets image placeholder rules, pure mode gets YAML frontmatter preservation rules

### Fixed

- **URL Processors**: Respect `--pure`/`--llm`/`--keep-base` flags for base `.md` output in both single and batch URL processing
- **Pure Mode Frontmatter**: `process_with_llm` uses `clean_document_pure()` instead of `process_document()` in pure mode, preventing LLM-generated frontmatter (description, tags, etc.)
- **Source Frontmatter Reconstruction**: Reconstruct original YAML frontmatter from defuddle metadata before sending to LLM in pure mode
- **Vision Prompt Drift**: Add placeholder REMINDER to vision prompts to reduce LLM drift on `__MARKITAI_IMG_N__` placeholders
- **Stabilization Dedup**: Deduplicate stabilization calls and add `paged_stabilized` guard
- **Vision JSON Mode**: Fix wrong message index in vision `json_mode` and race condition in parallel gather
- **Misc Fixes**: Frontmatter regex, env variable quoting, Ctrl+C handling, hardcoded weight, docstring corrections
- **SVG as Image-Only**: Treat SVG as image-only format in batch mode

### Changed

- **Output Strategy**: LLM mode skips writing base `.md` by default (use `--keep-base` to override)
- **Test Performance**: Optimize test suite speed (~70s → ~30s)

## [0.10.0] - 2026-03-12

### Added

- **Auto-detect LLM Providers**: When no `markitai.json` config exists, automatically detect available providers from environment variables and authenticated CLI tools (Claude CLI, Copilot CLI, Gemini CLI, ChatGPT OAuth)
- **Shared Provider Detection**: Extract provider detection into `cli/providers_detect.py` shared module for reuse across interactive and non-interactive modes

### Changed

- **Interactive Mode UX**: Separate OCR and screenshots from LLM features into independent "Additional options" prompt, since they are local processing capabilities (RapidOCR, Playwright) that don't require LLM
- **Feature Display**: Unified `build_feature_str()` in `ui.py` separates LLM features from local features with `|` delimiter (e.g., `LLM alt desc | OCR screenshot`)
- **Interactive Mode Flow**: Show configured models after user confirms LLM enablement, not before; warn when no provider detected
- **Dependencies**: Raise minimum constraints to match tested versions (pymupdf4llm >=1.27.2, litellm >=1.82.0, pydantic >=2.12.0, pytest >=9.0.0, ruff >=0.15.0)
- **CLI Flags**: `-v` is now `--verbose` (was `--version`), `-V` is now `--version`

### Fixed

- **Image Alt Text Language**: Strip YAML frontmatter before extracting document context for image analysis, so alt text matches the document's actual language instead of defaulting to English
- **Interactive Provider Display**: Show actual configured models from config file instead of auto-detected provider name
- **URL Processor Feature Display**: Add missing OCR to URL processor dry-run features list
- **Cold Startup Performance**: Lazy imports in `cli/`, `processors/`, and `workflow/` `__init__.py` reduce cold startup from ~5s to ~0.3s

### Removed

- **Language Field**: Remove LLM-generated `language` field from Frontmatter model — LLM should only generate `description` and `tags`, not infer extra metadata

## [0.9.2] - 2026-03-11

### Fixed

- **Copilot/Claude Login**: Revert subprocess output interception for copilot/claude-agent login — always use inherited stdio so the CLI sees a real TTY, fixing credential storage failures
- **Login Output Display**: Detect URL and device code on the same line (copilot outputs both together); track externally-printed lines for clean erasure after login
- **Error Message Clarity**: Fix `format_error_message` following `__context__` (implicit exception chain) to wrapper exceptions like tenacity `RetryError`, replacing informative provider errors with opaque `<Future at 0x...>` messages in logs; now only follows `__cause__` (explicit `raise X from Y`)
- **Error Message Consistency**: Use `format_error_message` in CLI catch-all handlers (`file.py`, `workflow/core.py`) to prevent opaque chained exception messages reaching users

### Added

- `SubprocessInterceptor` URL+code same-line formatting for copilot device code flow
- `OutputManager.track_external_lines()` for tracking terminal output from inherited-stdio subprocesses

## [0.9.1] - 2026-03-09

### Fixed

- **Provider Auth Preflight**: Add `can_attempt_login()` guard to skip login prompt when provider SDK is missing; fix Rich markup swallowing `[gemini-cli]` via `escape()`; fix "Login failed: Login failed:" duplication
- **Install Scripts Extras Parsing**: Fix greedy regex (`\[.*\]` → `\[[^]]*\]`) that captured TOML outer brackets, corrupting extras names like `gemini-cli}]`
- **Install Scripts Resilience**: Progressive fallback when full extras install fails (retry without SDK-dependent extras); fix `set -e` silent exit on `uv tool install` failure; fix PowerShell 5.x `Join-Path` 3-arg incompatibility
- **Install Scripts Extras Strategy**: Merge-based finalize (no longer replaces manually tracked extras); generic receipt parsing (future-proof for new extras)

### Added

- `markitai doctor --suggest-extras` as single source of truth for install scripts to query recommended extras
- `can_attempt_login()` provider guard with `get_auth_resolution_hint()` fallback messages
- i18n key `not_found` for zh-CN and en in both setup scripts

## [0.9.0] - 2026-03-09

### Added

- **Fetch Strategy Priority**: Configurable global and per-domain strategy ordering via `strategy_priority` in `policy` and `domain_profiles`
- **Domain/IP Exemption**: `local_only_patterns` config field restricts specified domains/IPs to local-only strategies (static, playwright) — supports exact domain, suffix (`.internal.com`), wildcard (`*.internal.com`), IP, and CIDR notation (`10.0.0.0/8`, `fd00::/8`)
- **NO_PROXY Integration**: `inherit_no_proxy` (default: true) automatically merges `NO_PROXY` environment variable patterns into local-only exemptions
- **Fetch Security Feature**: README documentation for the new information security compliance capabilities

### Fixed

- **LLM Language Consistency**: Strengthened 5 prompt templates to prevent language translation when fetching mixed-language content (e.g., English UI + Chinese body) — LLM now determines output language from body text, not UI elements

## [0.8.1] - 2026-03-06

### Added

- **Defuddle Fetch Strategy**: New `defuddle` strategy (`GET https://defuddle.md/<url>`) as top-priority URL fetch method — free, no auth, returns clean Markdown with YAML frontmatter (title, author, published, description, word_count, domain)
- **Aggressive Strategy Ordering**: Default ordering changed to `defuddle → jina → static → playwright → cloudflare` (both default and SPA scenarios)
- **CLI `--defuddle` Flag**: Force defuddle-only URL fetching (mutually exclusive with `--playwright`, `--jina`, `--cloudflare`)
- **DefuddleConfig**: Configurable timeout and RPM rate limiting (conservative defaults for undocumented API limits)

### Changed

- **FetchPolicyEngine**: Simplified ordering logic — removed `has_jina_key` branching; defuddle+jina always first
- **max_strategy_hops**: Default increased from 4 to 5 to accommodate the new strategy

## [0.8.0] - 2026-03-06

### Added

- **Extended Format Support**: 20+ new file formats via markitdown and kreuzberg converters
  - **Markitdown-based**: HTML/HTM/XHTML, CSV, EPUB, MSG, IPYNB (Jupyter Notebook), Apple Numbers
  - **Kreuzberg-based** (optional dependency): TSV, XML, ODS, ODT, SVG, RTF, RST, ORG, TEX, EML
  - Kreuzberg is a pure Rust wheel — install with `uv pip install markitai[kreuzberg]`
- **Extended Image Support**: GIF, BMP, TIFF now supported by ImageConverter; BMP/TIFF auto-converted to PNG for LLM vision APIs
- **LLM Vision Format Helpers**: `is_llm_supported_image()`, `get_llm_effective_mime()` in `utils/mime.py` for transparent BMP/TIFF → PNG handling

### Fixed

- **Claude Agent SDK v0.1.46 compatibility**: Removed deprecated `allow_dangerously_skip_permissions` parameter (`permission_mode="bypassPermissions"` is sufficient)
- **i18n test isolation**: Fixed global state leak in `test_i18n.py` causing 3 integration tests to fail when run in full suite
- **Import-time log leakage**: Kreuzberg registration logs changed from `logger.debug` to `logger.trace` to prevent terminal noise before CLI log setup

### Changed

- **Converter registry**: New `FileFormat` enum members for all added formats; kreuzberg registers as gap-filler (only for formats without native converters)
- **Test fixtures**: Renamed to consistent `sample.*` naming convention; added fixtures for all new formats; removed orphaned `sample.mobi`
- **Markitdown lazy init**: `MarkItDown()` in `markitdown_ext.py` now initialized on first use instead of import time

## [0.7.0] - 2026-03-05

### Added

- **ChatGPT Provider** (`chatgpt/`): Subscription-based provider using ChatGPT OAuth Device Code Flow and Responses API. No extra SDK required — uses LiteLLM's built-in authenticator. Models: `chatgpt/gpt-5.2`, `chatgpt/codex-mini`, etc.
- **Gemini CLI Provider** (`gemini-cli/`): Uses Google's Gemini CLI OAuth credentials (`~/.gemini/oauth_creds.json`) with automatic token refresh. Optional SDK: `uv add markitai[gemini-cli]`. Models: `gemini-cli/gemini-2.5-pro`, `gemini-cli/gemini-2.5-flash`, etc.
- **Weight=0 Model Disabling**: Setting `weight: 0` in model config now explicitly disables the model (excluded from routing). Useful for temporarily disabling models without removing config.
- **Interactive Mode Enhancements**: Updated onboarding wizard with ChatGPT and Gemini CLI provider options

### Fixed

- **ZeroDivisionError in Router**: Models with `weight=0` are now filtered before LiteLLM Router creation, preventing `division by zero` in `simple-shuffle` routing strategy when all selected models have zero weight
- **Router Weight Selection**: `_select_model` fallback uses `random.choice()` instead of `random.uniform(0, 0)` when all models have zero weight

### Changed

- **Weight Field Semantics**: `weight` field description updated to clarify that 0 = disabled. Minimum value enforced at 0 (negative weights rejected by validation)

## [0.6.1] - 2026-03-05

### Fixed

- **Claude Agent SDK compliance**: Add `allow_dangerously_skip_permissions=True` when using `bypassPermissions`, pass system messages via SDK's `system_prompt` parameter instead of XML tags, set `additionalProperties: false` in JSON object schema
- **Auth pre-check gaps**: Detect `GH_TOKEN`/`GITHUB_TOKEN` env vars as valid Copilot authentication, detect `CLAUDE_CODE_USE_BEDROCK`/`VERTEX`/`FOUNDRY` env vars as valid Claude authentication
- **Resolution hints**: Include env var alternatives in authentication error messages

### Changed

- **Docs**: Update configuration guide and ai-tools-setup with env var auth methods

## [0.6.0] - 2026-03-04

### Added

- **Cloudflare Integration**: Unified cloud backend with two capabilities:
  - **Browser Rendering**: `--cloudflare` flag for cloud-based URL rendering via CF `/markdown` API, with rate limiting, cache TTL, and advanced params (`user_agent`, `cookies`, `wait_for_selector`, `http_credentials`)
  - **Workers AI toMarkdown**: Cloud-based document conversion for PDF/XLSX/DOCX/PPTX (converter backend)
- **Fetch Policy Engine** (`fetch_policy.py`): Policy-driven strategy ordering with domain-specific profiles, session persistence, and adaptive targeting
- **Domain Profiles**: Per-domain fetch config (`wait_for_selector`, `wait_for`, `extra_wait_ms`, `prefer_strategy`) in `markitai.json`
- **Playwright Session Persistence**: `session_mode` (`isolated`/`domain_persistent`) and `session_ttl_seconds` for reusing browser contexts across requests
- **Static HTTP Abstraction** (`fetch_http.py`): Pluggable HTTP backend with `httpx` (default) and `curl-cffi` (TLS fingerprint impersonation) via `MARKITAI_STATIC_HTTP` env var
- **Content Validation Gate**: All fetch strategies now validate content quality before accepting results
- **`api_base` env: syntax**: `"api_base": "env:MY_BASE_URL"` in model config for environment variable expansion
- **CF Markdown for Agents**: Content negotiation via `Accept: text/markdown` header for Cloudflare-enabled sites

### Changed

- **Vision Router Fallback**: When all vision models are disabled (`weight=0`), falls back to main router with warning instead of crashing
- **Playwright UTF-8 Encoding**: Force UTF-8 for HTML-to-Markdown conversion to prevent encoding errors
- **Integration Test Resilience**: Cloudflare integration tests now skip on rate limit (429) instead of failing

### Fixed

- **ZeroDivisionError in Vision Router**: Models with `weight=0` (disabled) are now filtered out before litellm Router creation, preventing `division by zero` in `simple-shuffle` routing strategy
- **Dead Code Cleanup**: Removed 21 dead functions/classes across 15+ files (backward compat aliases, deprecated functions, unused constants)

### Removed

- `_html_to_text`, `_normalize_bypass_list`, `_get_proxy_bypass`, `get_proxy_for_url`, `_url_to_session_id` from `fetch.py`
- `sanitize_error_message` from `security.py`
- `_deep_update`, `get_config` from `config.py`
- `order_dict_keys_sorted`, `_order_image_entry` from `json_order.py`
- `reset_consoles` from `console.py`
- `get_llm_not_configured_hint` from `hints.py`
- `remove_uncommented_screenshots`, `_UNCOMMENTED_SCREENSHOT_RE` from `llm/content.py`
- `get_pending_urls`, `finish_url_processing` from `batch.py`
- `LLMUsageAccumulator` from `workflow/helpers.py`
- `DEFAULT_LOG_PANEL_MAX_LINES` from `constants.py`
- Multiple backward-compatibility aliases from `cli/processors/`

## [0.5.2] - 2026-02-07

### Fixed

- **SQLite ResourceWarning**: Close SQLite connections explicitly via `_connect()` context manager, preventing `ResourceWarning: unclosed database` on Python 3.13
- **Windows path handling**: `context_display_name()` now handles `C:/` forward-slash Windows paths (was only handling `C:\`)
- **Windows install hints**: `markitai doctor` shows platform-specific install commands (PowerShell/winget on Windows, curl on Unix)
- **OAuth token expiry**: `markitai doctor` no longer reports "Token expired" when a valid refresh token exists
- **Config get output**: `markitai config get` renders Pydantic models as formatted JSON with syntax highlighting instead of raw Python repr
- **Copilot ProviderError**: Added missing `provider` kwarg when raising `ProviderError` for unsupported models
- **Pyright warnings**: Resolved all Pyright warnings (lazy `__all__`, type narrowing, optional imports)

### Changed

- **26 documentation fixes**: Comprehensive audit fixing docstring-to-code mismatches across all modules (llm, providers, converter, utils, config)

## [0.5.1] - 2026-02-07

### Added

- **Playwright auto-scroll**: Auto-scroll pages to trigger lazy-loaded content before extraction (up to 8 steps, inspired by baoyu-skills url-to-markdown)
- **DOM noise cleanup**: Remove navigation, ads, cookie banners, popups, and inline event handlers before content extraction
- **`python -m markitai`**: Add `__main__.py` for `-m` invocation support (fixes Windows execution)
- **Multi-provider detection**: Interactive mode (`-I`) now detects and displays all available LLM providers (DeepSeek, OpenRouter included)
- **Copilot GPT-5 series support**: GPT-5, GPT-5.1, GPT-5.2, GPT-5.1-Codex-Mini/Max, GPT-5.2-Codex now fully supported via Copilot provider
- **22 new unit tests**: Vision fallback strategies, smart_truncate edge cases, content protection roundtrip, cache fingerprint collision resistance, batch thread safety

### Changed

- **Default models modernized**: Updated outdated defaults across init/interactive/doctor (haiku→sonnet, gpt-4o→gpt-5.2, gemini-2.0→2.5, claude-sonnet-4→4.5)
- **Init wizard**: Multi-provider default selection, API keys stored in `.env` instead of plaintext config, next-steps hints after completion
- **LLM code deduplication**: `document.py` now delegates `_protect_image_positions` / `_restore_image_positions` to `content.py` shared functions
- **Cache fingerprint**: SHA256 over full content + page structure replaces `text[:1000]` prefix-based cache keys, preventing collisions for documents with identical prefixes
- **Batch thread safety**: Double-checked locking with timeout-based lock acquisition (5s) replaces non-blocking `acquire(blocking=force)`
- **LiteLLM model database**: Refreshed with 35 new models including Claude Opus 4.6

### Fixed

- **DOM cleanup JS syntax error**: Selectors with double quotes (e.g., `[role="banner"]`) now properly escaped via `json.dumps()` instead of f-string interpolation
- **Copilot model blocklist**: Removed outdated GPT-5 series from `UNSUPPORTED_MODELS` (only o1/o3 reasoning models remain blocked)
- **CLI provider display**: Truncate provider list with `(+N more)` when >3 detected to prevent line overflow

## [0.5.0] - 2026-02-06

### Added

- **Unified UI system**: New `ui.py` components and `i18n.py` module with Chinese/English support across all CLI commands
- **`markitai init`**: One-stop setup wizard — checks dependencies, detects LLM providers, generates config
- **Interactive mode** (`-I`): Guided setup with questionary prompts for new users
- **`doctor --fix`**: Auto-install missing components (e.g., Playwright)
- **Cross-platform install hints**: Platform-specific installation commands in doctor output
- **`MARKITAI_LOG_FORMAT`**: Environment variable override for log format
- **JSON repair**: Fallback parser for malformed LLM JSON responses using `json_repair`

### Changed

#### Performance

- **CLI startup**: Lazy-load processor and command modules (~3x faster `--help`)
- **Dependency checks**: Parallelized doctor and init with `ThreadPoolExecutor`
- **LLM processing**: Pre-compiled regex patterns and batched replacements
- **PDF rendering**: Parallel page rendering for standard and LLM modes
- **URL fetching**: Async-safe cache locking for concurrent requests
- **Executor**: Auto-detect heavy task limit based on system RAM
- **Image processing**: Offloaded CPU-intensive work to thread pool
- **Cache stats**: Merged stats and model breakdown into single SQLite query

#### Refactoring

- **Batch UI**: Replaced Rich table/LogPanel with compact unified UI (progress bar with current file, completion summary)
- **Log format**: Default changed to human-readable text (was JSON)
- **LLM cache**: Deduplicated `SQLiteCache`/`PersistentCache` into `llm/cache.py`
- **Single file output**: Layered output with `--verbose` for detailed logs
- **Setup scripts**: Consolidated 10 scripts into 2 unified files (`setup.sh` + `setup.ps1`) with built-in i18n

### Fixed

- **Windows**: LibreOffice detection with fallback to `Program Files` paths (not just PATH)
- **Windows**: FFmpeg/CLI path display — show "installed" instead of long winget package paths
- **Windows**: `config path` alignment with dynamic padding and continuous `│` column
- **Playwright**: Default `wait_for` changed to `domcontentloaded` (was `networkidle`, caused hangs)
- **Config**: Schema and function defaults synced with constants
- **Exceptions**: Preserved exception chains (`raise from`) across codebase
- **Cache**: Prevented stale `markitai_processed` timestamp on cache hit
- **CLI**: Version flag reverted to `-v/--version`, `--verbose` kept without short flag

### CI

- Added Windows LibreOffice install step (`choco`) to CI matrix
- Changed to `--all-extras` for comprehensive dependency testing
- Publish workflow: split unit/integration tests with `SKIP_LLM_TESTS`

## [0.4.2] - 2026-02-03

### Changed

- **Playwright defaults**: `wait_for` changed to `networkidle`, `extra_wait_ms` to 5000ms for better SPA support
- **Frontmatter validation**: Pydantic validators reject empty description/tags, triggering Instructor auto-retry
- **VitePress**: Upgraded to 2.0.0-alpha.16

### Fixed

- **X/Twitter content**: Pages now wait for full JS rendering before capture
- **Cache directories**: All caches now respect `cache.global_dir` config instead of hardcoded paths
- **Setup scripts**: Improved piped execution (`curl | sh`), proper Playwright installation paths
- **Config init**: Added `--yes/-y` flag for non-interactive use

## [0.4.1] - 2026-02-02

### Added

- **`markitai doctor`**: New diagnostic command for system health and auth status checking
- **Adaptive timeout**: Local providers auto-adjust timeout based on request complexity
- **Prompt caching**: Claude Agent caches long system prompts (>4KB) for cost reduction

### Changed

- `check-deps` renamed to `doctor` (old name kept as alias)
- Improved error messages with resolution hints for local providers

### Fixed

- Request timeouts on large documents with Claude Agent / Copilot
- JSON extraction issues with control characters and markdown code blocks

## [0.4.0] - 2026-01-28

### Added

- **Claude Agent SDK**: `claude-agent/sonnet|opus|haiku` via Claude Code CLI
- **GitHub Copilot SDK**: `copilot/claude-sonnet-4.5|gpt-4o|o1` models
- **URL HTTP caching**: ETag/Last-Modified conditional requests
- **Quiet mode**: `--quiet` / `-q` flag (auto-enabled for single file)
- **Module refactoring**: `cli.py` → `cli/`, `llm.py` → `llm/`, new `providers/`
- **Setup scripts hardening**: default N for high-impact ops, version pinning
- **Docs**: CONTRIBUTING.md, architecture.md, ai-tools-setup.md, dependabot.yml

### Changed

- Python 3.13, docs reorganized to `docs/archive/`
- agent-browser locked to 0.7.6 (Windows bug in 0.8.x)
- Default `extra_wait_ms`: 1000 → 3000, Instructor mode: `JSON` → `MD_JSON`

### Fixed

- **Windows**: UTF-8 console, Copilot CLI path discovery, script argument quoting
- **LLM**: Frontmatter regex fallback, `source` field fix, vision/frontmatter error handling
- **Prompts**: Enhanced prompt leakage prevention, placeholder protection rules
- **Content**: Social media cleanup rules (X/Twitter, Facebook, Instagram)
- **Setup**: WSL detection, Python pymanager support, PATH refresh order

## [0.3.2] - 2026-01-27

### Added

- Chinese README (`README_ZH.md`) with language toggle
- Chinese setup scripts: `setup-zh.sh`, `setup-zh.ps1`, `setup-dev-zh.sh`, `setup-dev-zh.ps1`

### Changed

- Improved setup scripts with better error handling and user feedback
- Updated Python version note: 3.11-3.13 (3.14 not yet supported)
- Updated documentation language toggle links

## [0.3.1] - 2026-01-27

### Fixed

#### Prompt Leakage Prevention
- Split all prompts into `*_system.md` (role definition) and `*_user.md` (content template)
- Added `_validate_no_prompt_leakage()` to detect and handle prompt leakage in LLM output
- Updated LLM calls to use proper `[{"role": "system"}, {"role": "user"}]` message structure

#### LLM Compatibility
- Fixed `max_tokens` exceeding deepseek limit by using minimum across all router models
- Fixed terminal window popup on Windows when running agent-browser verification

#### URL Fetching
- Improved error messages for browser fetch timeout (no longer suggests installing when already attempted)
- Added auto-proxy detection for Jina API and browser fetching
  - Checks environment variables: `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`
  - Auto-detects local proxy ports: 7890 (Clash), 10808 (V2Ray), 1080 (SOCKS5), etc.

### Added

#### SPA Domain Learning
- New `SPADomainCache` for automatic detection and caching of JavaScript-heavy sites
- `markitai cache spa-domains` command to view/manage learned domains
- `markitai cache clear --include-spa-domains` option

#### Windows Performance Optimizations
- Thread pool optimization: Windows defaults to 4 workers (vs 8 on Linux/macOS)
- ONNX Runtime global singleton with preheat for OCR engine
- OpenCV-based image compression (releases GIL, 20-40% faster)
- Batch subprocess execution for agent-browser commands

### Changed

- Default image quality: 85 → 75
- Default image max_height: 1080 → 99999 (effectively unlimited)
- Default image min_area filter: 2500 → 5000
- Default URL concurrency: 3 → 5
- Default scan_max_depth: 10 → 5
- Extended fallback_patterns with more social media domains

## [0.3.0] - 2026-01-26

### Added

#### URL Conversion Support
- **Direct URL conversion**: `markitai <url>` converts web pages to Markdown
- **URL batch processing**: Support `.urls` file format (text or JSON), auto-detected from input
- **URL image downloading**: `download_url_images()` with concurrent downloads (5 parallel)
- Automatic relative URL resolution for images
- Cross-platform filename sanitization (Windows illegal characters handling)

#### Multi-Source URL Fetching (fetch.py)
- **Three fetch strategies**: `--static` / `--agent-browser` / `--jina`
  - `static`: MarkItDown direct HTTP fetch (default, fastest)
  - `browser`: agent-browser headless rendering (for JS-heavy pages)
  - `jina`: Jina Reader API (cloud-based, no local deps)
  - `auto`: Smart fallback (static → browser/jina if JS detected)
- **FetchCache**: SQLite-based URL cache with LRU eviction (100MB default)
- **Screenshot capture**: `--screenshot` for full-page screenshots via browser
- **Multi-source content**: Parallel static + browser fetch with quality validation
- Domain pattern matching for auto-browser fallback (x.com, twitter.com, etc.)
- `FetchResult` with `static_content`, `browser_content`, `screenshot_path`

#### agent-browser Integration
- Headless browser automation via `agent-browser` CLI
- Configurable wait states: `load`, `domcontentloaded`, `networkidle`
- Extra wait time for SPA rendering (`extra_wait_ms`)
- Session isolation for concurrent fetches
- `verify_agent_browser_ready()` with cached readiness check
- Screenshot compression with Pillow (JPEG quality + max height)

#### URL LLM Enhancement
- New `prompts/url_enhance.md` for URL-specific content cleaning
- Multi-source LLM processing: combine static + browser + screenshot
- Smart content selection based on validity detection

#### Cache Enhancements
- **`--no-cache-for <pattern>`**: Selective cache bypass with glob patterns
  - Single file: `--no-cache-for file1.pdf`
  - Glob pattern: `--no-cache-for "*.pdf"`
  - Mixed: `--no-cache-for "*.pdf,reports/**"`
- **`markitai cache stats -v`**: Verbose mode with detailed cache entries
- **`--limit N`**: Control number of entries in verbose output (default: 20)
- **`--scope project|global|all`**: Filter cache statistics by scope
- `SQLiteCache.list_entries()`: List cache entries with metadata
- `SQLiteCache.stats_by_model()`: Per-model cache statistics
- Improved cache hash: head + tail + length algorithm for better invalidation

#### Workflow Core Refactor (workflow/core.py)
- **`ConversionContext`**: Unified single-file conversion context
- **`convert_document_core()`**: Main conversion pipeline
  - `validate_and_detect_format()` → `convert_document()` → `process_embedded_images()`
  - `write_base_markdown()` → `process_with_vision_llm()` / `process_with_standard_llm()`
- Parallel document + image processing with proper dependency handling
- Alt text injection after LLM processing completes (race condition fix)

#### Official Website
- VitePress 2.x documentation site with bilingual support (English/Chinese)
- Custom theme with brand colors matching logo
- Local search integration
- GitHub Actions auto-deployment to GitHub Pages

#### Project
- **MIT License**: Added LICENSE file

#### CI/CD
- **`.github/workflows/ci.yml`**: Automated testing on push/PR
- **`.github/workflows/deploy-website.yml`**: Website deployment to GitHub Pages

#### Code Architecture
- New `utils/paths.py`: `ensure_dir()`, `ensure_subdir()`, `ensure_assets_dir()`
- New `utils/mime.py`: `get_mime_type()`, `get_extension_from_mime()`
- New `utils/text.py`: `normalize_markdown_whitespace()`, text utilities
- New `utils/executor.py`: `run_in_executor()` with shared ThreadPoolExecutor
- New `utils/output.py`: Output formatting helpers
- New `json_order.py`: Ordered JSON serialization for reports/state files
- New `urls.py`: `.urls` file parser (JSON and plain text formats)
- `LLMUsageAccumulator` class for centralized cost tracking
- `create_llm_processor()` factory function
- Unified `detect_language()` with `get_language_name()` helper
- Centralized `IMAGE_EXTENSIONS`, `JS_REQUIRED_PATTERNS` constants

#### Configuration
- **`supports_vision` now optional**: Auto-detected from litellm when not explicitly set
  - No need to manually configure for most models (GPT-4o, Gemini, Claude, etc.)
  - Explicit `supports_vision: true/false` overrides auto-detection if needed

### Changed

#### Package Rename
- **`markit` → `markitai`**: Package renamed for clarity
- CLI command remains `markitai`

#### Python Version
- **Python 3.11+ support**: Lowered minimum Python version from 3.13 to 3.11

#### CLI Behavior
- **Single file mode**: Direct stdout output (no logging by default)
- **`--verbose`**: Show logs before output in single file mode
- Batch processing behavior unchanged

#### Code Quality
- Refactored PowerShell COM conversion scripts (~18% code reduction)
- Unified MIME type mapping across codebase
- Extracted common fixtures to `conftest.py`
- Improved error messages for network failures (SSL/connection/proxy)
- Architecture diagram updated in `docs/spec.md`

### Fixed
- URL filename cross-platform compatibility
- Cache invalidation for large documents (tail changes now detected)
- Image analysis race condition with `.llm.md` file writing

## [0.2.4] - 2026-01-21

### Changed
- Restructured `assets.json` format with flat asset array
- Extract Live display management for early log capture
- Improved MS Office detection with file path fallback

### Fixed
- Add openpyxl FileVersion compatibility patch
- Add pptx XMLSyntaxError compatibility patch
- Enhanced `check_symlink_safety` with nested symlink detection
- LLM empty response retry logic
- `normalize_frontmatter` for consistent YAML field order

## [0.2.3] - 2026-01-20

### Added

#### Persistent LLM Cache
- SQLite-based cache with LRU eviction and size limits (default 1GB)
- Dual-layer lookup: project cache + global cache
- `CacheConfig` in `MarkitaiConfig` with enabled/no_cache/max_size options
- **`--no-cache` CLI flag**: Skip reading but still write (Bun semantics)
- **`markitai cache stats [--json]`**: View cache statistics
- **`markitai cache clear [--scope]`**: Clear cache by scope

#### Vision Router Optimization
- Smart router selection: auto-detect image content in messages
- `vision_router` property filtering only `supports_vision=true` models
- Replace hardcoded "vision" model name with "default" + smart routing

#### Legacy Office Conversion
- MS Office COM batch conversion: one app launch per file type
- `check_ms_word/excel/powerpoint_available()` registry-based detection
- Pre-convert legacy files before batch processing to reduce overhead

#### Performance (Phase 3)
- **Parallel PDF processing**: Concurrent page OCR & rendering
- **Parallel image processing**: `ProcessPoolExecutor` for CPU-bound compression
- Adaptive worker count based on file size
- LRU eviction and byte-size limits for image cache
- Batch semaphore for memory pressure control

### Changed
- OCR optimization: `recognize_numpy()` and `recognize_pixmap()` for direct array processing
- Reuse already-rendered pixmap in PDF OCR (avoid re-rendering)

### Fixed
- EMF/WMF format detection and PNG conversion support
- `DATA_URI_PATTERN` regex for hyphenated MIME types (x-emf, x-wmf)
- Base64 stripping: remove hallucinated images instead of replacing
- Batch timing: record `start_at` before pre-conversion for accurate duration
- Pyright venv detection: add venvPath/venv to pyproject.toml

## [0.2.2] - 2026-01-20

### Added
- `constants.py` module to consolidate hardcoded values
- Unit tests for image and llm modules
- `convert_to_markdown.py` reference script

### Changed
- Centralized constants usage across config.py, llm.py, batch.py, image.py
- Improved LLM content restoration with garbage detection logic
- Enable parallel batch processing for image analysis
- Move state saving outside semaphore to reduce blocking

### Fixed
- Rich Panel markup parsing issue (escape file paths)

## [0.2.1] - 2026-01-20

### Added

#### LLM Usage Tracking
- Context-based usage tracking (per-file instead of global)
- `get_context_cost()` and `get_context_usage()` for per-file stats
- Thread-safe lock for concurrent access to usage dictionaries

#### Type System
- `types.py` with TypedDict definitions (ModelUsageStats, LLMUsageByModel, AssetDescription)
- `ImageAnalysis.llm_usage` for multi-model tracking (renamed from `model`)

#### Model Configuration
- `get_model_max_output_tokens()` using litellm.get_model_info()
- Auto-inject max_tokens with fallback to conservative default (8192)

#### Office Detection
- `utils/office.py` module with cross-platform detection
- `has_ms_office()`: Windows COM-based MS Office detection
- `find_libreoffice()`: PATH + common paths search with `@lru_cache`

#### Image Processing
- `strip_base64_images()` method
- `remove_nonexistent_images()` to clean LLM-hallucinated references
- Normalize whitespace for standalone image `.llm.md` output

### Changed
- File conflict rename strategy: `.2.md` → `.v2.md` for natural sort order
- Batch state: add `screenshots` field (separate from embedded images)
- Batch state: add `log_file` field for run traceability
- Store file paths as relative to input_dir in batch state

## [0.2.0] - 2026-01-19

### Added
- **Monorepo architecture** with uv workspace (`packages/markitai/`)
- **LiteLLM integration** for unified LLM provider access
- New converter modules: `pdf`, `office`, `image`, `text`, `legacy`
- Workflow system for single file processing (`workflow/single.py`)
- Markdown-based prompt management system (`prompts/*.md`)
- Unified config with JSON schema validation (`config.schema.json`)
- Security module for path validation (`security.py`)
- Comprehensive test suite with fixtures

### Changed
- CLI rewritten with Click (replaced Typer)
- Requires Python 3.13+

### Removed
- Old `src/markitai/` structure and all legacy code
- Complex pipeline/router/state machine architecture
- Individual LLM provider implementations (OpenAI, Anthropic, etc.)
- Docker and CI scripts (to be re-added later)

### Breaking Changes
- Configuration format changed (see migration guide)
- CLI command syntax updated
- Python 3.12 and below no longer supported

## [0.1.6] - 2026-01-14

### Fixed
- Model routing strategy bugs
- Documentation accuracy improvements

## [0.1.5] - 2026-01-13

### Changed
- Refactored prompt management system for better maintainability
- Simplified cleaner module logic

## [0.1.4] - 2026-01-13

### Fixed
- JSON parsing edge cases in LLM responses
- Log formatting improvements for readability

## [0.1.3] - 2026-01-12

### Added
- Test coverage improved to 81%

### Changed
- Adopted `src` layout for project structure
- Reorganized documentation to `docs/reference/`
- Added GitHub Actions CI workflow

### Fixed
- Provider-specific bugs in fallback handling

## [0.1.2] - 2026-01-12

### Added
- Resilience features for network failures (retry logic, timeout handling)
- `CLAUDE.md` and `AGENTS.md` documentation for AI assistants

### Changed
- Log optimization for cleaner, more informative output

## [0.1.1] - 2026-01-11

### Changed
- Major architecture refactoring with service layer pattern
- Enhanced LLM support with better error handling and retries

## [0.1.0] - 2026-01-10

### Added

#### Capability-Based Model Routing
- `required_capability` and `prefer_capability` parameters for LLM calls
- Text tasks prioritize text-only models for cost efficiency
- Vision tasks automatically use vision-capable models
- Backward compatible: parameters default to None (round-robin behavior)

#### Lazy Model Initialization
- Providers loaded on-demand instead of all at startup
- Significantly reduced initialization time for single-file conversions
- `warmup()` method for batch mode to validate providers upfront
- `required_capabilities` parameter in `initialize()`

#### Concurrent Fallback Mechanism
- Primary model timeout triggers parallel backup model execution
- Neither model is interrupted - first response wins
- Configurable via `llm.concurrent_fallback_timeout` (default: 180s)
- Handles Gemini 504 timeout scenarios gracefully

#### Execution Mode Support
- `--fast` flag for speed-optimized batch processing
- Fast mode: skips validation, limits fallback attempts, reduces logging
- Default mode: full validation, detailed logging, comprehensive retries
- Configurable via `execution.mode` in config file

#### Enhanced Statistics
- `BatchStats` class for comprehensive processing metrics
- Per-model tracking: calls, tokens, duration, estimated cost
- `ModelCostConfig` for optional cost estimation
- Summary format: "Complete: X success, Y failed | Total: Xs | Tokens: N"

### Changed
- CLI architecture refactored for better modularity
- Config format migrated from JSON to YAML

## [0.0.1] - 2026-01-08

### Added
- **Initial release**
- CLI commands: `convert`, `batch`, `config`, `provider`
- Multi-format support: Word (.doc, .docx), PowerPoint (.ppt, .pptx), Excel (.xls, .xlsx), PDF, HTML
- LLM enhancement: markdown formatting, frontmatter generation, image alt text
- 5 LLM providers with fallback: OpenAI, Anthropic, Gemini, Ollama, OpenRouter
- 3 PDF engines: pymupdf4llm (default), pymupdf, pdfplumber
- Image processing: extraction, compression (oxipng/mozjpeg), LLM analysis
- Batch processing with resume capability and concurrency control
- Unit and integration tests
- Docker multi-stage build
- Chinese and English documentation

[0.12.1]: https://github.com/Ynewtime/markitai/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/Ynewtime/markitai/compare/v0.11.2...v0.12.0
[0.11.2]: https://github.com/Ynewtime/markitai/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/Ynewtime/markitai/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/Ynewtime/markitai/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Ynewtime/markitai/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/Ynewtime/markitai/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/Ynewtime/markitai/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/Ynewtime/markitai/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/Ynewtime/markitai/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/Ynewtime/markitai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Ynewtime/markitai/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Ynewtime/markitai/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Ynewtime/markitai/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/Ynewtime/markitai/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/Ynewtime/markitai/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Ynewtime/markitai/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/Ynewtime/markitai/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/Ynewtime/markitai/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Ynewtime/markitai/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/Ynewtime/markitai/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Ynewtime/markitai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Ynewtime/markitai/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/Ynewtime/markitai/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/Ynewtime/markitai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Ynewtime/markitai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Ynewtime/markitai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Ynewtime/markitai/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/Ynewtime/markitai/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/Ynewtime/markitai/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/Ynewtime/markitai/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/Ynewtime/markitai/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Ynewtime/markitai/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Ynewtime/markitai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Ynewtime/markitai/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Ynewtime/markitai/releases/tag/v0.0.1
