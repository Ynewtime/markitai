/** en/zh UI dictionaries — CLI tone: verb-first, no hype, tech terms stay
 * English in zh (mirrors the docs site register). Status words themselves
 * ("queued", "converting", "done", "failed") are CLI output and stay English
 * in both locales — see STATUS_TEXT in components/ItemRow.tsx. In zh prose
 * the proper noun is always uppercase "LLM"; en control labels stay lowercase
 * (CLI voice). Visible copy never uses em/en dashes; "·" is the separator
 * voice (the "…" in masked keys is an ellipsis, not a dash).
 */

export type Locale = "en" | "zh";

const LANG_KEY = "markitai.lang";

/** Stored explicit choice ("en"/"zh") wins; anything else means auto. */
export function detectLocale(): Locale {
  try {
    const stored = localStorage.getItem(LANG_KEY);
    if (stored === "en" || stored === "zh") return stored;
  } catch {
    /* localStorage unavailable */
  }
  const lang = typeof navigator === "undefined" ? "en" : navigator.language;
  return lang.toLowerCase().startsWith("zh") ? "zh" : "en";
}

export function storeLocale(locale: Locale): void {
  try {
    localStorage.setItem(LANG_KEY, locale);
  } catch {
    /* localStorage unavailable */
  }
}

const en = {
  heroTitle: "Drop files. Get Markdown.",
  heroSub:
    "Convert documents, images, and web pages to clean Markdown. Add LLM enhancement only when you need it.",
  urlPlaceholder: "paste URLs · one per line · enter to convert",
  // ≤780px the full line wraps and the 1-row textarea clips it — shorter copy
  urlPlaceholderShort: "paste URLs · enter to convert",
  convert: "convert",
  preset: "preset",
  llmEnhance: "llm enhance",
  dropAnywhere: "or drop files anywhere on this page",
  browse: "browse",
  dropMore: "drop files to convert more",
  dropToConvert: "drop to convert",
  converting: "Converting",
  done: "Done",
  statDone: "done",
  statFailed: "failed",
  statSkipped: "skipped",
  conversions: "conversions",
  currentSession: "current session",
  clearAll: "clear all",
  clearCompleted: "clear completed",
  nothingCompleted: "no completed jobs to clear",
  downloadAllZip: "download all (.zip)",
  downloadJobZip: "download this job (.zip)",
  zipWhileRunning: "available when this job finishes",
  downloadMd: "download .md",
  rendered: "rendered",
  source: "source",
  copy: "copy",
  copied: "copied",
  words: "words",
  loading: "loading…",
  capHintPre: "llm not configured · ",
  capHintLink: "configure llm",
  capHintPost: " to enable standard/rich presets",
  createJobFailed: "create job failed",
  skipImageOnly: "image input · enable llm or ocr",
  skipExists: "output already exists",
  skipCfgPre: "image input · ",
  skipCfgLink: "configure llm",
  skipCfgPost: " or enable ocr",
  colName: "Name",
  colTime: "Time",
  colCost: "Cost",
  colStatus: "Status",
  total: "Total",
  itemsAria: "converted items",
  previewAria: "preview mode",
  themeAria: "theme",
  langAria: "language",
  settingsAria: "llm settings",
  settingsTitle: "llm settings",
  breadcrumbAria: "settings path",
  addModels: "add models",
  addModelsCount: (n: number) => `add ${n} models`,
  modelsAdded: (n: number) => `${n} models added`,
  deploymentsConfigured: (n: number) => `${n} configured deployment${n === 1 ? "" : "s"}`,
  detectedSession: "detected for this session",
  saveToConfig: "save to config",
  loadModels: "load models",
  refreshModels: "refresh models",
  modelCatalogTitle: "available models",
  connectProviderTitle: (label: string) => `connect ${label}`,
  modelCatalogLoading: "loading available models…",
  modelsUnavailable: "models could not be loaded",
  modelsPartial: "some models may be missing",
  providerCardMeta: (kind: string, status: string, source: string) => {
    if (status === "disabled") return "disabled";
    if (kind === "configured") return "saved connection";
    if (kind === "local_cli") return status === "ready" ? "signed in" : "sign in required";
    if (kind === "oauth") return status === "ready" ? "OAuth connected" : "sign in required";
    if (kind === "environment") return `using ${source}`;
    if (status === "needs_credentials") return "credentials required";
    return status === "unknown" ? "not checked" : status.replaceAll("_", " ");
  },
  providerDetailHint: (kind: string, label: string, source: string) => {
    if (kind === "configured") return "Available models load automatically from this saved connection.";
    if (kind === "local_cli") return `Models available to your signed-in ${label} account.`;
    if (kind === "oauth") return `Models available to your connected ${label} account.`;
    if (kind === "environment") return `Using credentials from ${source}.`;
    if (label === "Ollama") return "Using the default local Ollama endpoint.";
    return "Enter the connection details to load available models.";
  },
  customApiBase: "use a custom api base",
  requiredField: "required",
  manualModelToggle: "add a model ID manually",
  providerGroup: (kind: string) =>
    ({
      local_cli: "detected local CLI",
      oauth: "detected OAuth",
      environment: "environment credentials",
      configured: "configured connections",
      common: "common providers",
    })[kind] ?? kind,
  providerStatus: (status: string) => status.replaceAll("_", " "),
  searchModels: "search models",
  selectVisible: "select visible",
  manualModelPh: "full model id",
  azureDeploymentPh: "Azure deployment name",
  addManual: "add manually",
  modelsAvailable: "available models",
  alreadyConfigured: "configured",
  noModelsFound: "no models found · enter one manually",
  advanced: "advanced",
  routingGroup: "routing group",
  weight: "weight",
  modelsSelected: (n: number, max: number) => `${n} selected · ${max} max`,
  confirmDeleteModel: (model: string) => `confirm permanent deletion of ${model}`,
  deleteModel: (model: string) => `permanently delete ${model}`,
  setStatusNone: "no llm configured",
  setSourceLbl: "source",
  setModel: "model",
  setModelName: "model name",
  setApiKey: "api key",
  setApiBase: "api base",
  setModelNamePh: "unique name · e.g. default",
  setModelPh: "provider/model-id · e.g. deepseek/deepseek-chat",
  setKeyPh: "sk-… or env:OPENAI_API_KEY",
  providerKeyPh: (provider: string) =>
    `sk-… or env:${
      ({
        anthropic: "ANTHROPIC_API_KEY",
        azure: "AZURE_API_KEY",
        deepseek: "DEEPSEEK_API_KEY",
        gemini: "GEMINI_API_KEY",
        openrouter: "OPENROUTER_API_KEY",
      })[provider] ?? "OPENAI_API_KEY"
    }`,
  setBasePh: "optional",
  setKeyHint: "written to the config file · use env:VAR to reference an environment variable",
  keepKeyHint: "leave blank to keep current key",
  keepBaseHint: "leave blank to keep current api base",
  localNote: "local provider - no API key needed",
  addModel: "add model",
  cancel: "cancel",
  edit: "edit",
  detDetected: (label: string) => `${label} detected`,
  detAdd: "add",
  show: "show",
  hide: "hide",
  test: "test",
  testing: "testing…",
  save: "save",
  saving: "saving…",
  saved: "saved",
  close: "close",
  homeAria: "markitai home",
  historyAria: "view conversions",
  histTitle: "recent conversions",
  histSubtitle: "finished jobs stored on this server",
  histEmpty: "no recent conversions",
  histRefreshing: "refreshing…",
  retryLoad: "retry",
  deleting: "deleting…",
  histToday: "today",
  histYesterday: "yesterday",
  histOpen: "open",
  histDelete: "delete",
  histConfirm: "delete permanently?",
  histDeleteAria: (name: string) => `permanently delete ${name}`,
  histConfirmDeleteAria: (name: string) => `confirm permanent deletion of ${name}`,
  histDeleted: (name: string) => `${name} permanently deleted`,
  histStorageSize: (size: string) => `storage ${size}`,
  histMore: (n: number) => `+${n} more`,
  sessResults: (n: number) =>
    `${n} ${n === 1 ? "item" : "items"} in session · view results`,
  sessProgress: (n: number) => `${n} converting · view progress`,
  themeAuto: "auto",
  themeLight: "light",
  themeDark: "dark",
  opensNewTab: "(opens in new tab)",
  srcAria: "markdown source",
  codeAria: "code block",
  tableAria: "table",
  ariaSeconds: "seconds",
  errExpandTitle: "click to show the full error",
  announceItem: (name: string, status: string, settled: number, total: number) =>
    `${name} ${status} · ${settled} of ${total} complete`,
  figureFrom: (name: string) => `figure from ${name}`,
  cliToggle: "copy as CLI command",
  cliAria: "CLI command",
  cliHint: "replace <your-files> with your local file paths",
  retryAria: (name: string) => `retry ${name}`,
  retryFailed: "retry failed",
  dropTruncated: (kept: number, total: number) =>
    `${kept} of ${total} files added (job limit)`,
  dropEmptyFolder: "no convertible files in the dropped folder",
  filterPh: "filter...",
  filterAria: "filter items by name",
  filterStatusAria: "filter by status",
  filterShown: (n: number, m: number) => `${n} of ${m} shown`,
  filterNoMatch: "no items match the filter",
  diffTab: "diff",
  diffTooLarge: "document too large to diff",
  diffAria: "line diff of base and llm markdown",
};

export type Dict = typeof en;

const zh: Dict = {
  heroTitle: "拖入文件，得到 Markdown。",
  heroSub: "转换文档、图片和网页为干净的 Markdown，需要时再开启 LLM 增强。",
  urlPlaceholder: "粘贴 URL · 每行一个 · enter 转换",
  urlPlaceholderShort: "粘贴 URL · enter 转换",
  convert: "转换",
  preset: "preset",
  llmEnhance: "LLM 增强",
  dropAnywhere: "或将文件拖到页面任意位置",
  browse: "浏览文件",
  dropMore: "拖入文件继续转换",
  dropToConvert: "松手开始转换",
  converting: "转换中",
  done: "完成",
  statDone: "完成",
  statFailed: "失败",
  statSkipped: "跳过",
  conversions: "转换任务",
  currentSession: "当前会话",
  clearAll: "全部清空",
  clearCompleted: "清除已完成",
  nothingCompleted: "没有可清除的已完成任务",
  downloadAllZip: "下载全部 (.zip)",
  downloadJobZip: "下载此任务 (.zip)",
  zipWhileRunning: "此任务完成后可下载",
  downloadMd: "下载 .md",
  rendered: "渲染",
  source: "源码",
  copy: "复制",
  copied: "已复制",
  words: "词",
  loading: "加载中…",
  capHintPre: "LLM 未配置 · ",
  capHintLink: "配置 LLM",
  capHintPost: " 后可用 standard/rich preset",
  createJobFailed: "创建任务失败",
  skipImageOnly: "图片输入 · 启用 LLM 或 OCR",
  skipExists: "输出文件已存在",
  skipCfgPre: "图片输入 · ",
  skipCfgLink: "配置 LLM",
  skipCfgPost: " 或启用 OCR",
  colName: "名称",
  colTime: "耗时",
  colCost: "成本",
  colStatus: "状态",
  total: "合计",
  itemsAria: "转换结果",
  previewAria: "预览模式",
  themeAria: "主题",
  langAria: "语言",
  settingsAria: "LLM 设置",
  settingsTitle: "LLM 设置",
  breadcrumbAria: "设置路径",
  addModels: "添加 model",
  addModelsCount: (n: number) => `添加 ${n} 个 model`,
  modelsAdded: (n: number) => `已添加 ${n} 个 model`,
  deploymentsConfigured: (n: number) => `已配置 ${n} 个 deployment`,
  detectedSession: "本次会话检测到",
  saveToConfig: "保存到配置",
  loadModels: "加载 model",
  refreshModels: "刷新 model",
  modelCatalogTitle: "可用 model",
  connectProviderTitle: (label: string) => `连接 ${label}`,
  modelCatalogLoading: "正在加载可用 model…",
  modelsUnavailable: "无法加载 model",
  modelsPartial: "部分 model 可能未显示",
  providerCardMeta: (kind: string, status: string, source: string) => {
    if (status === "disabled") return "已停用";
    if (kind === "configured") return "已保存连接";
    if (kind === "local_cli") return status === "ready" ? "已登录" : "需要登录";
    if (kind === "oauth") return status === "ready" ? "OAuth 已连接" : "需要登录";
    if (kind === "environment") return `使用 ${source}`;
    if (status === "needs_credentials") return "需要凭据";
    return status === "unknown" ? "尚未检查" : status.replaceAll("_", " ");
  },
  providerDetailHint: (kind: string, label: string, source: string) => {
    if (kind === "configured") return "已从保存的连接自动加载可用 model。";
    if (kind === "local_cli") return `显示当前 ${label} 登录账户可用的 model。`;
    if (kind === "oauth") return `显示当前 ${label} 连接账户可用的 model。`;
    if (kind === "environment") return `使用 ${source} 中的凭据。`;
    if (label === "Ollama") return "使用本机默认 Ollama 地址。";
    return "填写连接信息后加载可用 model。";
  },
  customApiBase: "使用自定义 api base",
  requiredField: "必填",
  manualModelToggle: "手动添加 model ID",
  providerGroup: (kind: string) =>
    ({
      local_cli: "检测到的本地 CLI",
      oauth: "检测到的 OAuth",
      environment: "环境变量凭据",
      configured: "已配置连接",
      common: "常用 provider",
    })[kind] ?? kind,
  providerStatus: (status: string) => status.replaceAll("_", " "),
  searchModels: "搜索 model",
  selectVisible: "选择当前结果",
  manualModelPh: "完整 model ID",
  azureDeploymentPh: "Azure deployment 名称",
  addManual: "手动添加",
  modelsAvailable: "可用 model",
  alreadyConfigured: "已配置",
  noModelsFound: "没有找到 model · 可手动输入",
  advanced: "高级",
  routingGroup: "routing group",
  weight: "权重",
  modelsSelected: (n: number, max: number) => `已选 ${n} 个 · 上限 ${max} 个`,
  confirmDeleteModel: (model: string) => `确认永久删除 ${model}`,
  deleteModel: (model: string) => `永久删除 ${model}`,
  setStatusNone: "未配置 LLM",
  setSourceLbl: "来源",
  setModel: "model",
  setModelName: "名称",
  setApiKey: "api key",
  setApiBase: "api base",
  setModelNamePh: "唯一名称 · 如 default",
  setModelPh: "provider/model-id · 如 deepseek/deepseek-chat",
  setKeyPh: "sk-… 或 env:OPENAI_API_KEY",
  providerKeyPh: (provider: string) =>
    `sk-… 或 env:${
      ({
        anthropic: "ANTHROPIC_API_KEY",
        azure: "AZURE_API_KEY",
        deepseek: "DEEPSEEK_API_KEY",
        gemini: "GEMINI_API_KEY",
        openrouter: "OPENROUTER_API_KEY",
      })[provider] ?? "OPENAI_API_KEY"
    }`,
  setBasePh: "可选",
  setKeyHint: "写入配置文件 · 可用 env:VAR 引用环境变量",
  keepKeyHint: "留空保持当前 key",
  keepBaseHint: "留空保持当前 api base",
  localNote: "本地 provider - 无需 API key",
  addModel: "添加 model",
  cancel: "取消",
  edit: "编辑",
  detDetected: (label: string) => `检测到 ${label}`,
  detAdd: "添加",
  show: "显示",
  hide: "隐藏",
  test: "测试",
  testing: "测试中…",
  save: "保存",
  saving: "保存中…",
  saved: "已保存",
  close: "关闭",
  homeAria: "markitai 首页",
  historyAria: "查看转换任务",
  histTitle: "最近转换",
  histSubtitle: "保存在此服务器上的已完成任务",
  histEmpty: "暂无最近转换",
  histRefreshing: "刷新中…",
  retryLoad: "重试",
  deleting: "删除中…",
  histToday: "今天",
  histYesterday: "昨天",
  histOpen: "打开",
  histDelete: "删除",
  histConfirm: "永久删除?",
  histDeleteAria: (name: string) => `永久删除 ${name}`,
  histConfirmDeleteAria: (name: string) => `确认永久删除 ${name}`,
  histDeleted: (name: string) => `${name} 已永久删除`,
  histStorageSize: (size: string) => `存储 ${size}`,
  histMore: (n: number) => `+${n} 项`,
  sessResults: (n: number) => `会话中有 ${n} 个结果 · 查看结果`,
  sessProgress: (n: number) => `${n} 个转换中 · 查看进度`,
  themeAuto: "跟随系统",
  themeLight: "浅色",
  themeDark: "深色",
  opensNewTab: "(在新标签页打开)",
  srcAria: "Markdown 源码",
  codeAria: "代码块",
  tableAria: "表格",
  ariaSeconds: "秒",
  errExpandTitle: "点击展开完整错误",
  announceItem: (name: string, status: string, settled: number, total: number) =>
    `${name} ${status} · 已完成 ${settled}/${total}`,
  figureFrom: (name: string) => `来自 ${name} 的图片`,
  cliToggle: "复制为 CLI 命令",
  cliAria: "CLI 命令",
  cliHint: "把 <your-files> 替换为本地文件路径",
  retryAria: (name: string) => `重试 ${name}`,
  retryFailed: "重试失败",
  dropTruncated: (kept: number, total: number) =>
    `已添加 ${kept}/${total} 个文件 (任务上限)`,
  dropEmptyFolder: "文件夹中没有可转换的文件",
  filterPh: "筛选...",
  filterAria: "按名称筛选",
  filterStatusAria: "按状态筛选",
  filterShown: (n: number, m: number) => `显示 ${n}/${m}`,
  filterNoMatch: "没有匹配的条目",
  diffTab: "对比",
  diffTooLarge: "文档过大，无法对比",
  diffAria: "base 与 LLM Markdown 的行级对比",
};

export const dicts: Record<Locale, Dict> = { en, zh };
