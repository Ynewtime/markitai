/** en/zh UI dictionaries — concise, verb-first, no hype; English UI uses
 * sentence case and tech terms stay English in zh. Internal status tokens
 * remain English in both locales — see STATUS_TEXT in ItemRow.tsx. In zh prose
 * the proper noun is always uppercase "LLM"; technical acronyms keep their
 * standard casing. Visible copy never uses em/en dashes; "·" is the separator
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
  heroTitle: "Drop files. Paste URLs. Get Markdown.",
  heroSub:
    "Turn documents, images, and web pages into clean Markdown. Add LLM enhancement only when needed.",
  urlPlaceholder: "Paste URLs · one per line · or drop files",
  // ≤780px the full line wraps and the 1-row textarea clips it — shorter copy
  urlPlaceholderShort: "Paste URLs or drop files",
  convert: "Convert",
  preset: "Preset",
  llmEnhance: "LLM enhancement",
  // ≤780px the full label pushes the options row past one line — the switch
  // keeps the full name in its aria-label (same split as urlPlaceholderShort)
  llmEnhanceShort: "LLM",
  llmEnhancedLabel: "LLM enhanced",
  ocr: "OCR",
  browse: "Choose files",
  dropToConvert: "Drop to convert",
  converting: "Converting",
  done: "Done",
  statDone: "Done",
  statFailed: "Failed",
  statSkipped: "Skipped",
  notifyBody: (done: number, failed: number) =>
    failed > 0 ? `${done} done · ${failed} failed` : `${done} done`,
  conversions: "Conversions",
  currentSession: "Current session",
  sessionBadge: "Session",
  clearAll: "Clear all",
  clearCompleted: "Clear completed",
  nothingCompleted: "No completed jobs to clear",
  emptyWorkspace: "No conversions yet - drop files or paste URLs above.",
  downloadAllZip: "Download all (.zip)",
  downloadingZip: "Preparing .zip…",
  downloadFailed: "Archive download failed",
  zipWhileRunning: "Available when all conversions finish",
  downloadMd: "Download .md",
  exportPdf: "Export PDF",
  pdfSettings: "PDF settings",
  pdfCustomHeaderFooter: "Custom header and footer",
  pdfPrintDialogHint: "When enabled, turn off the browser's own headers and footers in the print dialog.",
  pdfPreparedBy: "Prepared with markitai",
  pdfSource: "Source",
  rendered: "Rendered",
  source: "Source",
  copy: "Copy",
  copied: "Copied",
  copyFailed: "Copy failed",
  words: "Words",
  loading: "Loading…",
  capHintPre: "LLM not configured · ",
  capHintLink: "Configure LLM",
  capHintPost: " to enable enhancement",
  createJobFailed: "Create job failed",
  connLost: "Lost connection to the server",
  skipImageOnly: "Image skipped because neither LLM enhancement nor OCR was enabled.",
  skipExists: "Output already exists",
  imageSkippedTitle: "Image skipped",
  imageSkipped: (name: string) => `${name} needs LLM enhancement or OCR to be converted.`,
  enableOcr: "Enable OCR and retry",
  colName: "Name",
  colDuration: "Duration",
  colFinished: "Finished ↓",
  colCost: "LLM / Cost",
  colStatus: "Status",
  total: "Total",
  itemsAria: "Converted items",
  previewAria: "Preview mode",
  themeAria: "Theme",
  langAria: "Language",
  // the modal hosts appearance controls too (phone tier), not just LLM
  settingsAria: "Settings",
  settingsTitle: "Settings",
  appearanceTitle: "Appearance",
  breadcrumbAria: "Settings path",
  addModels: "Add models",
  addModelsCount: (n: number) => `Add ${n} models`,
  modelsAdded: (n: number) => `${n} models added`,
  modelsConfigured: (n: number) => `${n} model${n === 1 ? "" : "s"} configured`,
  providersTitle: "Saved providers",
  providerModels: (n: number) => `${n} model${n === 1 ? "" : "s"}`,
  selectProvider: (label: string) => `Select ${label}`,
  editProvider: (label: string) => `Edit ${label} provider`,
  deleteProvider: (label: string) => `Delete ${label} provider`,
  deleteProviderTitle: (label: string) => `Delete ${label}?`,
  deleteProviderDescription: (n: number) =>
    n > 0
      ? `This clears its credentials and removes ${n} configured model${n === 1 ? "" : "s"}.`
      : "This permanently clears the saved credentials.",
  providerConnectionMissing:
    "This saved provider connection no longer exists. Refresh the provider list and choose it again.",
  providerSaved: "Provider saved",
  detectedSession: "Detected for this session",
  saveToConfig: "Save to config",
  loadModels: "Load models",
  refreshModels: "Refresh models",
  modelCatalogTitle: "Available models",
  connectProviderTitle: (label: string) => `Connect ${label}`,
  modelCatalogLoading: "Loading available models…",
  modelsUnavailable: "Models could not be loaded",
  modelsPartial: "Some models may be missing",
  providersLoadFailed: "Could not load providers",
  providerCardMeta: (kind: string, status: string, source: string) => {
    if (status === "disabled") return "Disabled";
    if (status === "missing_dependency") return "Runtime support missing";
    if (kind === "configured") return "Saved connection";
    if (kind === "local_cli") return status === "ready" ? "Signed in" : "Sign in required";
    if (kind === "oauth") return status === "ready" ? "OAuth connected" : "Sign in required";
    if (kind === "environment") return `Using ${source}`;
    if (status === "needs_credentials") return "Credentials required";
    if (status === "unknown") return "Not checked";
    const label = status.replaceAll("_", " ");
    return label.charAt(0).toUpperCase() + label.slice(1);
  },
  providerDetailHint: (kind: string, label: string, source: string) => {
    if (kind === "configured") return "Available models load automatically from this saved connection.";
    if (kind === "local_cli") return `Models available to your signed-in ${label} account.`;
    if (kind === "oauth") return `Models available to your connected ${label} account.`;
    if (kind === "environment") return `Using credentials from ${source}.`;
    if (label === "Ollama") return "Using the default local Ollama endpoint.";
    return "Enter the connection details to load available models.";
  },
  customApiBase: "Use a custom API base",
  requiredField: "Required",
  manualModelToggle: "Add a model ID manually",
  providerGroup: (kind: string) =>
    ({
      local_cli: "Detected local CLI",
      oauth: "Detected OAuth",
      environment: "Environment credentials",
      configured: "Saved providers",
      common: "Common providers",
    })[kind] ?? kind,
  searchModels: "Search models",
  selectVisible: "Select visible",
  manualModelPh: "Full model ID",
  azureDeploymentPh: "Azure deployment name",
  addManual: "Add manually",
  modelsAvailable: "Available models",
  alreadyConfigured: "Configured",
  noModelsFound: "No models found · enter one manually",
  advanced: "Advanced",
  routingGroup: "Routing group",
  weight: "Routing weight",
  modelWeight: (n: number) => `Routing weight ${n}`,
  // phone-width pill — the weightHint title keeps the full explanation
  modelWeightShort: (n: number) => `Weight ${n}`,
  weightHint: "Routing weight used when models share the same routing group",
  modelsSelected: (n: number, max: number) => `${n} selected · ${max} max`,
  deleteModel: (model: string) => `Permanently delete ${model}`,
  deleteModelTitle: (model: string) => `Delete ${model}?`,
  deleteModelDescription: "The provider credentials will be kept for adding models later.",
  setStatusNone: "No models configured",
  setSourceLbl: "Source",
  setModel: "Model",
  setApiKey: "API key",
  setApiBase: "API base",
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
  setBasePh: "Optional",
  revealField: (label: string) => `Show ${label}`,
  concealField: (label: string) => `Hide ${label}`,
  cancel: "Cancel",
  edit: "Edit",
  test: "Test",
  testing: "Testing…",
  modelTestPassed: "Model test passed",
  modelTestReady: (model: string) => `${model} is ready to use.`,
  modelTestFailed: "Model test failed",
  save: "Save",
  saving: "Saving…",
  saved: "Saved",
  settingsConflict: "Settings were changed in another window - reload and try again",
  close: "Close",
  homeAria: "Markitai home",
  historyAria: "View conversions",
  historyCurrent: "Conversion tasks · current page",
  docsLabel: "Docs",
  retryLoad: "Retry",
  jobLoadFailed: "Could not load this job",
  nothingToPreview: "This job has no previewable result",
  deleting: "Deleting…",
  histOpen: "Open",
  histDeleteAria: (name: string) => `Permanently delete ${name}`,
  deleteItemTitle: (name: string) => `Delete ${name}?`,
  deleteItemDescription: "This removes the stored result and cannot be undone.",
  deletePermanently: "Delete permanently",
  histDeleted: (name: string) => `${name} permanently deleted`,
  histStorageSize: (size: string) => `Storage ${size}`,
  histMore: (n: number) => `+${n} more`,
  sessResults: (n: number) =>
    `${n} ${n === 1 ? "item" : "items"} in session · View results`,
  sessProgress: (n: number) => `${n} converting · View progress`,
  themeAuto: "Auto",
  themeLight: "Light",
  themeDark: "Dark",
  opensNewTab: "(Opens in new tab)",
  srcAria: "Markdown source",
  codeAria: "Code block",
  tableAria: "Table",
  ariaSeconds: "Seconds",
  errExpandTitle: "Click to show the full error",
  announceItem: (name: string, status: string, settled: number, total: number) =>
    `${name} ${status} · ${settled} of ${total} complete`,
  figureFrom: (name: string) => `Figure from ${name}`,
  cliToggle: "Copy as CLI command",
  cliAria: "CLI command",
  cliHint: "Replace <your-files> with your local file paths",
  retryAria: (name: string) => `Retry ${name}`,
  retryFailed: "Retry failed",
  enhanceWithLlm: (name: string) => `Enhance ${name} with LLM`,
  llmEnhanceUnavailable: "Configure a routable LLM in settings to use this action",
  llmEnhanceTurnOn: "Turn on LLM enhancement above to use this action",
  llmEnhanceFailed: "LLM enhancement failed",
  noFailedItem: "No retryable item remains in this job",
  noEnhanceableItem: "No base Markdown result remains in this job",
  dropTruncated: (kept: number, total: number) =>
    `${kept} of ${total} files added (job limit)`,
  dropEmptyFolder: "No convertible files in the dropped folder",
  filterPh: "Filter…",
  filterAria: "Filter items by name",
  filterStatusAria: "Filter by status",
  filterShown: (n: number, m: number) => `${n} of ${m} shown`,
  filterNoMatch: "No items match the filter",
  diffTab: "Diff",
  diffTooLarge: "Document too large to diff",
  diffAria: "Line diff of base and LLM Markdown",
};

export type Dict = typeof en;

const zh: Dict = {
  heroTitle: "拖入文件，粘贴 URL，得到 Markdown。",
  heroSub: "将文档、图片和网页转换为干净的 Markdown，需要时再开启 LLM 增强。",
  urlPlaceholder: "粘贴 URL（每行一个），或拖入文件转换",
  urlPlaceholderShort: "粘贴 URL，或拖入文件",
  convert: "转换",
  preset: "预设",
  llmEnhance: "LLM 增强",
  llmEnhanceShort: "LLM",
  llmEnhancedLabel: "LLM 增强",
  ocr: "OCR",
  browse: "选择文件",
  dropToConvert: "松手开始转换",
  converting: "转换中",
  done: "完成",
  statDone: "完成",
  statFailed: "失败",
  statSkipped: "跳过",
  notifyBody: (done: number, failed: number) =>
    failed > 0 ? `完成 ${done} · 失败 ${failed}` : `完成 ${done}`,
  conversions: "转换任务",
  currentSession: "当前会话",
  sessionBadge: "本次会话",
  clearAll: "全部清空",
  clearCompleted: "清除已完成",
  nothingCompleted: "没有可清除的已完成任务",
  emptyWorkspace: "还没有转换任务——在上方拖入文件或粘贴 URL。",
  downloadAllZip: "下载全部 (.zip)",
  downloadingZip: "正在准备 .zip…",
  downloadFailed: "归档下载失败",
  zipWhileRunning: "全部转换完成后可下载",
  downloadMd: "下载 .md",
  exportPdf: "导出 PDF",
  pdfSettings: "PDF 设置",
  pdfCustomHeaderFooter: "自定义页眉页脚",
  pdfPrintDialogHint: "启用后，请在打印对话框中关闭浏览器自带的页眉页脚。",
  pdfPreparedBy: "由 markitai 生成",
  pdfSource: "来源",
  rendered: "渲染",
  source: "源码",
  copy: "复制",
  copied: "已复制",
  copyFailed: "复制失败",
  words: "词",
  loading: "加载中…",
  capHintPre: "LLM 未配置 · ",
  capHintLink: "配置 LLM",
  capHintPost: " 后可启用增强",
  createJobFailed: "创建任务失败",
  connLost: "与服务器的连接已断开",
  skipImageOnly: "图片已跳过，因为未启用 LLM 增强或 OCR。",
  skipExists: "输出文件已存在",
  imageSkippedTitle: "图片已跳过",
  imageSkipped: (name: string) => `${name} 需要启用 LLM 增强或 OCR 才能转换。`,
  enableOcr: "启用 OCR 并重试",
  colName: "名称",
  colDuration: "耗时",
  colFinished: "完成时间 ↓",
  colCost: "LLM / 成本",
  colStatus: "状态",
  total: "合计",
  itemsAria: "转换结果",
  previewAria: "预览模式",
  themeAria: "主题",
  langAria: "语言",
  settingsAria: "系统设置",
  settingsTitle: "系统设置",
  appearanceTitle: "外观",
  breadcrumbAria: "设置路径",
  addModels: "添加模型",
  addModelsCount: (n: number) => `添加 ${n} 个模型`,
  modelsAdded: (n: number) => `已添加 ${n} 个模型`,
  modelsConfigured: (n: number) => `已配置 ${n} 个模型`,
  providersTitle: "已保存的服务商",
  providerModels: (n: number) => `${n} 个模型`,
  selectProvider: (label: string) => `选择 ${label}`,
  editProvider: (label: string) => `编辑 ${label} 服务商`,
  deleteProvider: (label: string) => `删除 ${label} 服务商`,
  deleteProviderTitle: (label: string) => `删除 ${label}？`,
  deleteProviderDescription: (n: number) =>
    n > 0
      ? `这会清除鉴权信息，并删除已配置的 ${n} 个模型。`
      : "这会永久清除已保存的鉴权信息。",
  providerConnectionMissing: "找不到已保存的服务商连接，请刷新列表后重新选择。",
  providerSaved: "服务商已保存",
  detectedSession: "本次会话检测到",
  saveToConfig: "保存到配置",
  loadModels: "加载模型",
  refreshModels: "刷新模型",
  modelCatalogTitle: "可用模型",
  connectProviderTitle: (label: string) => `连接 ${label}`,
  modelCatalogLoading: "正在加载可用模型…",
  modelsUnavailable: "无法加载模型",
  modelsPartial: "部分模型可能未显示",
  providersLoadFailed: "无法加载服务商列表",
  providerCardMeta: (kind: string, status: string, source: string) => {
    if (status === "disabled") return "已停用";
    if (status === "missing_dependency") return "缺少运行依赖";
    if (kind === "configured") return "已保存连接";
    if (kind === "local_cli") return status === "ready" ? "已登录" : "需要登录";
    if (kind === "oauth") return status === "ready" ? "OAuth 已连接" : "需要登录";
    if (kind === "environment") return `使用 ${source}`;
    if (status === "needs_credentials") return "需要凭据";
    return status === "unknown" ? "尚未检查" : status.replaceAll("_", " ");
  },
  providerDetailHint: (kind: string, label: string, source: string) => {
    if (kind === "configured") return "已从保存的连接自动加载可用模型。";
    if (kind === "local_cli") return `显示当前 ${label} 登录账户可用的模型。`;
    if (kind === "oauth") return `显示当前 ${label} 连接账户可用的模型。`;
    if (kind === "environment") return `使用 ${source} 中的凭据。`;
    if (label === "Ollama") return "使用本机默认 Ollama 地址。";
    return "填写连接信息后加载可用模型。";
  },
  customApiBase: "使用自定义 API 地址",
  requiredField: "必填",
  manualModelToggle: "手动添加模型 ID",
  providerGroup: (kind: string) =>
    ({
      local_cli: "检测到的本地 CLI",
      oauth: "检测到的 OAuth",
      environment: "环境变量凭据",
      configured: "已保存的服务商",
      common: "常用服务商",
    })[kind] ?? kind,
  searchModels: "搜索模型",
  selectVisible: "选择当前结果",
  manualModelPh: "完整模型 ID",
  azureDeploymentPh: "Azure 部署名称",
  addManual: "手动添加",
  modelsAvailable: "可用模型",
  alreadyConfigured: "已配置",
  noModelsFound: "没有找到模型，可手动输入",
  advanced: "高级",
  routingGroup: "路由组",
  weight: "路由权重",
  modelWeight: (n: number) => `路由权重 ${n}`,
  modelWeightShort: (n: number) => `权重 ${n}`,
  weightHint: "同一路由组有多个模型时，用于决定各模型被选中的比例",
  modelsSelected: (n: number, max: number) => `已选 ${n} 个 · 上限 ${max} 个`,
  deleteModel: (model: string) => `永久删除 ${model}`,
  deleteModelTitle: (model: string) => `删除 ${model}？`,
  deleteModelDescription: "服务商鉴权信息会保留，之后仍可添加模型。",
  setStatusNone: "未配置模型",
  setSourceLbl: "来源",
  setModel: "模型",
  setApiKey: "API Key",
  setApiBase: "API 地址",
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
  revealField: (label: string) => `显示${label}`,
  concealField: (label: string) => `隐藏${label}`,
  cancel: "取消",
  edit: "编辑",
  test: "测试",
  testing: "测试中…",
  modelTestPassed: "模型测试通过",
  modelTestReady: (model: string) => `${model} 已连接并可正常使用。`,
  modelTestFailed: "模型测试失败",
  save: "保存",
  saving: "保存中…",
  saved: "已保存",
  settingsConflict: "设置已在其他窗口被修改，请刷新后重试",
  close: "关闭",
  homeAria: "markitai 首页",
  historyAria: "查看转换任务",
  historyCurrent: "转换任务 · 当前页面",
  docsLabel: "文档",
  retryLoad: "重试",
  jobLoadFailed: "无法加载该任务",
  nothingToPreview: "该任务没有可预览的结果",
  deleting: "删除中…",
  histOpen: "打开",
  histDeleteAria: (name: string) => `永久删除 ${name}`,
  deleteItemTitle: (name: string) => `删除 ${name}？`,
  deleteItemDescription: "这会移除已保存的结果，且无法撤销。",
  deletePermanently: "永久删除",
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
  enhanceWithLlm: (name: string) => `使用 LLM 增强 ${name}`,
  llmEnhanceUnavailable: "请先在设置中配置可用的 LLM",
  llmEnhanceTurnOn: "请先打开上方的 LLM 增强开关",
  llmEnhanceFailed: "LLM 增强失败",
  noFailedItem: "此任务中没有可重试的条目",
  noEnhanceableItem: "此任务中没有可进行 LLM 增强的基础结果",
  dropTruncated: (kept: number, total: number) =>
    `已添加 ${kept}/${total} 个文件 (任务上限)`,
  dropEmptyFolder: "文件夹中没有可转换的文件",
  filterPh: "筛选…",
  filterAria: "按名称筛选",
  filterStatusAria: "按状态筛选",
  filterShown: (n: number, m: number) => `显示 ${n}/${m}`,
  filterNoMatch: "没有匹配的条目",
  diffTab: "对比",
  diffTooLarge: "文档过大，无法对比",
  diffAria: "基础版与 LLM 增强版 Markdown 的行级对比",
};

export const dicts: Record<Locale, Dict> = { en, zh };
