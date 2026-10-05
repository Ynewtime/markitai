// Service messages in the interface language. The service answers in English;
// a stable code (an API error's `reason`, then its status `code`; an item's
// `error_code`) or a recognizable message shape picks the localized text, and
// every result keeps the service's own wording as `detail` for diagnosis.
// A message that matches nothing is shown as written.
import type { Locale } from "./locale.ts";

const en = {
  errBadRequest: "The service could not accept this request.",
  errForbidden: "The service refused this request.",
  errNotFound: "The service could not find what was requested.",
  errConflict: "This request conflicts with the current state. Refresh and try again.",
  errBusy: "The service is busy. Try again in a moment.",
  errUnavailable: "The service is temporarily unavailable. Try again shortly.",
  errInternal: "The service hit an internal error. Try again; if it keeps happening, check the server log.",
  errTryAgain: "The service could not create the job this time. Submit it again.",
  errShuttingDown: "The service is shutting down and cannot accept new work.",
  errHostNotAllowed:
    "The service does not accept this address. Open it through localhost, an IP address, or a name allowed with --allowed-host.",
  errOriginNotAllowed: "The service refused a request coming from another site.",
  errSettingsForbidden: "Model settings can be changed only on this computer or with the access token.",
  errRequestTooLarge: "This upload is larger than the service accepts in one request.",
  errFileTooLarge: "A file is larger than the 100 MiB upload limit.",
  errTooManyItems: "This job has more files and URLs than the service accepts.",
  errInvalidUrlList: "That URL list could not be read. Use one URL per line, or a JSON array of URLs.",
  errEmptyUrlList: "That URL list holds no usable http:// or https:// URL.",
  errUrlListTooLarge: "That URL list is larger than the service accepts (1 MiB).",
  errInvalidOutputName: "A name in that URL list is not a plain filename. Remove any folder path from it.",
  errInvalidUrl: "One of the web addresses is not valid.",
  errUrlScheme: "Only http:// and https:// addresses can be converted.",
  errRemoteUrl: "Web pages can be converted only on this computer or with the access token. Files can still be uploaded.",
  errInvalidOptions: "The service did not accept these conversion options.",
  errRemoteProcessingForbidden: "This connection cannot request Cloudflare processing. Open the service with its access token.",
  errRemoteConfirmation: "Confirm the Cloudflare scope again before sending this request.",
  errRemoteDisabled: "The server’s policy disables remote processing. Choose the native reader or contact the administrator.",
  errCloudflareUnavailable: "Cloudflare is not ready on this server. Check the server’s Cloudflare configuration.",
  errInvalidRequest: "The service could not read this request.",
  errJobNotFound: "This job no longer exists. It may have been deleted.",
  errItemNotFound: "This item is no longer part of the job.",
  errJobRunning: "Wait until the job has finished, then try again.",
  errResultUnavailable: "This result is not available. Retry the item to create it again.",
  errResultTooLarge: "This result is too large to show here. Download the .zip to get it.",
  errFileNotFound: "This file is no longer on the server.",
  errArchiveTooLarge: "The archive would contain too many files. Download the results separately.",
  errHistoryEmpty: "There are no saved jobs to download yet.",
  errDownloadTicket: "The download link expired or was already used. Start the download again.",
  errTooManyTickets: "Too many downloads are being started at once. Wait a minute and try again.",
  errOutputConflict: "This item's output files overlap another item's, so the service left them unchanged.",
  errPersistenceFailed: "This job could not be saved. Restart the service to recover it before retrying.",
  errBatchPending: "A provider batch for this item is still pending. Collect it before retrying or enhancing.",
  errItemBusy: "This item is still being converted. Retry it when it has finished.",
  errNotRetryable: "Items recorded by the command line cannot be retried here. Run markitai on the file again.",
  errNoSource: "This item has no original file or address to convert again.",
  errUploadMissing: "The original upload is no longer on the server. Upload the file again.",
  errLlmUnavailable: "LLM processing needs a working model. Add one in Settings first.",
  errProviderBusy: "Too many model requests are running. Try again in a moment.",
  errProviderRequest: "The provider settings or environment variable reference are not valid.",
  errProviderUnavailable: "The provider request could not complete. Check the endpoint, API key and network.",
  errDiscoveryTimeout: "Model discovery timed out.",
  errSettingsInvalid: "The service did not accept this settings change.",
  errSettingsMissing: "This saved setting no longer exists. Refresh and try again.",
  errSettingsIo: "The configuration file could not be read or saved.",
  errConfigMissing: "There is no configuration file yet. Save a model to create it.",
  errSettingsReadOnly:
    "Settings are read-only because the service was started with model or provider overrides. Edit the configuration file and restart without them.",
  errStaleRevision: "Settings were changed in another window.",
  errConfigChanged: "The configuration file changed while saving. Reload settings and try again.",
  errAmbiguousModel: "More than one model uses this routing name. Edit the configuration file to make it unique.",
  errSettingsDurability:
    "The configuration was saved, but the service could not confirm it is safely on disk. Reload settings before another change.",
  errConfigTooLarge: "The configuration would exceed the 8 MiB limit.",
  errNoModel: "No model is configured. Add one in Settings, or set MODEL and a provider API key.",
  errOcrUnavailable: "Local OCR is not available on this system.",
  errOcrVisionSelection:
    "This setting selects macOS Vision, which requires macOS 11 or later. Unset MARKITAI_OCR_BACKEND or choose paddle, then restart the service.",
  errOcrBackendSetting: "Set MARKITAI_OCR_BACKEND to vision or paddle, or unset it, then restart the service.",
  errOcrModelMissing:
    "A local OCR model is missing. Run markitai doctor --fix on the service computer with network access, or follow the original error's verified manual-download instructions, then retry.",
  errOcrModelCorrupt:
    "A local OCR model is damaged. Run markitai doctor --fix on the service computer with network access to verify and replace it, then retry. Ordinary OCR leaves it unchanged.",
  errOcrModelPath:
    "The local OCR installation path is unsafe or changed during the operation. Run markitai doctor on the service computer and check the named path's ownership, permissions and links. Resolve the path issue before retrying; automatic repair does not fix unsafe paths.",
  errOcrModelDownload:
    "A local OCR model could not be downloaded. Check the service computer's network and proxy, then run markitai doctor --fix or follow the original error's verified manual-download instructions.",
  errOcrModelPreparation:
    "Local OCR model preparation failed. Run markitai doctor on the service computer and inspect the original error before retrying.",
  errUnsupported: "This conversion is not supported here.",
  errFetch: "Could not fetch this page.",
  errConversion: "The document could not be converted. Check that the original file opens correctly, then review the error details.",
  errInvalidInput: "The input was not accepted.",
  errConfig: "The service configuration does not allow this conversion.",
  errIo: "The service could not read or write a file.",
  errSourceMissing: "The source file could not be found.",
  errEnhanceNoResult: "LLM enhancement did not produce an enhanced version.",
  errNoOutput: "The conversion produced no output.",
  errTimeout: "The conversion took too long and was stopped.",
  errFetchTimeout: "Fetching this page took too long.",
  errModelFailed: "The model could not complete this document.",
  errUnreachable: "Could not connect to this website. Check the address and the server's network access.",
  errFetchPolicy: "Remote fetching is turned off in the service configuration.",
  errNoContent: "The page had no content that could be extracted.",
  errTooLarge: "The input is larger than a supported limit ({limit}).",
  errPageNotFound: "The page was not found (HTTP {status}).",
  errPageDenied: "The website refused access (HTTP {status}).",
  errPageRate: "The website is limiting requests (HTTP {status}). Try again later.",
  errPageServer: "The website had a server error (HTTP {status}).",
  errPageHttp: "The website answered HTTP {status}.",
  errModelTimeout: "The model did not respond in time.",
  errModelNotFound:
    "The model or its endpoint was not found (HTTP {status}). Check the model identifier and base URL.",
  errModelDenied: "The provider rejected the credentials (HTTP {status}). Check the API key.",
  errModelRate: "The provider is limiting requests (HTTP {status}). Try again later.",
  errModelServer: "The provider had a server error (HTTP {status}).",
  errModelHttp: "The provider answered HTTP {status}.",
  errModelRegion:
    "The provider does not offer this model in your region (HTTP {status}). Choose another provider or model.",
  errModelQuota: "The provider account has no remaining quota or needs billing set up (HTTP {status}).",
  errModelUnavailable: "The provider does not offer this model (HTTP {status}). Check the model identifier.",
  errHistoryNotSaved:
    "This job could not be saved to history. Its finished files stay available until the service stops.",
  errRollback: "Restoring the previous result failed. Restart the service to recover it.",
  needToken: "Not authorized · reload the page with the access token link",
  needInput: "Choose at least one file or enter a URL.",
  nothingToStop: "Nothing is waiting any more; the remaining items are already converting.",
  itemStopped: "Stopped before conversion · retry to convert it",
  itemShutdown: "Cancelled because the service stopped · retry to convert it",
  unsupportedFormat: "This file type is not supported: {format}.",
  probeOk: "{model} responded.",
  probeUnreachable: "Could not reach the model endpoint.",
  probeConfig: "The credentials or endpoint settings are invalid or missing.",
  probeUnsupported: "This provider is not supported by the built-in runtime.",
  probeFailed: "The connection test failed.",
  probeBadResponse: "The model endpoint returned a response that could not be read.",
  discoveryNeedsBase: "Enter an API base URL first.",
  discoveryNoRuntime: "This provider needs a sign-in or local runtime that is not available here.",
  discoveryFailed: "Model discovery failed. Check the endpoint, credentials and provider availability.",
  discoveryStaleShown: "Refresh failed; showing the models found earlier.",
  discoveryAzure: "Azure lists regional base models; routing still needs a deployment name.",
  discoveryPaged: "The provider has more models; only the first page is shown.",
  discoveryOfficial: "Models reported by the signed-in official {name} runtime.",
  discoveryOfficialMissing: "The official {name} runtime or its sign-in is not available.",
};

export type MessageKey = keyof typeof en;

const zh: Record<MessageKey, string> = {
  errBadRequest: "服务无法接受这个请求。",
  errForbidden: "服务拒绝了这个请求。",
  errNotFound: "服务找不到请求的内容。",
  errConflict: "请求与当前状态冲突。请刷新后重试。",
  errBusy: "服务繁忙，请稍后再试。",
  errUnavailable: "服务暂时不可用，请稍后再试。",
  errInternal: "服务内部出错。请重试；如果反复出现，请查看服务器日志。",
  errTryAgain: "这次未能创建任务，请重新提交。",
  errShuttingDown: "服务正在关闭，无法接收新的任务。",
  errHostNotAllowed: "服务不接受这个访问地址。请通过 localhost、IP 地址，或用 --allowed-host 允许的主机名访问。",
  errOriginNotAllowed: "服务拒绝了来自其他网站的请求。",
  errSettingsForbidden: "只有在本机或使用访问令牌时才能修改模型设置。",
  errRequestTooLarge: "本次上传超出服务单次请求可接收的大小。",
  errFileTooLarge: "有文件超过 100 MiB 的上传上限。",
  errTooManyItems: "这个任务包含的文件和 URL 超出服务上限。",
  errInvalidUrlList: "无法读取这个 URL 列表。请每行一个 URL，或使用 URL 的 JSON 数组。",
  errEmptyUrlList: "这个 URL 列表里没有可用的 http:// 或 https:// 地址。",
  errUrlListTooLarge: "这个 URL 列表超过服务上限（1 MiB）。",
  errInvalidOutputName: "URL 列表中的某个输出名不是纯文件名，请去掉其中的目录部分。",
  errInvalidUrl: "有网址无效。",
  errUrlScheme: "只能转换 http:// 或 https:// 地址。",
  errRemoteUrl: "只有在本机或使用访问令牌时才能转换网页；仍可上传文件。",
  errInvalidOptions: "服务不接受这些转换选项。",
  errRemoteProcessingForbidden: "此连接未获准请求 Cloudflare 处理；请使用访问令牌打开服务。",
  errRemoteConfirmation: "请重新确认 Cloudflare 的发送范围后再提交本次请求。",
  errRemoteDisabled: "服务器策略禁止远程处理；请选择原生读取或联系管理员。",
  errCloudflareUnavailable: "此服务器的 Cloudflare 尚未就绪，请检查服务器上的配置。",
  errInvalidRequest: "服务无法读取这个请求。",
  errJobNotFound: "这个任务已不存在，可能已被删除。",
  errItemNotFound: "任务中已没有这一项。",
  errJobRunning: "请等任务完成后再试。",
  errResultUnavailable: "这个结果不可用。请重试这一项以重新生成。",
  errResultTooLarge: "结果太大，无法在这里显示。请下载 .zip 获取。",
  errFileNotFound: "服务器上已没有这个文件。",
  errArchiveTooLarge: "压缩包包含的文件过多。请分别下载结果。",
  errHistoryEmpty: "还没有可下载的已保存任务。",
  errDownloadTicket: "下载链接已过期或已用过，请重新开始下载。",
  errTooManyTickets: "同时开始的下载过多，请稍等一分钟再试。",
  errOutputConflict: "这一项的输出文件与其他项重叠，服务没有改动它们。",
  errPersistenceFailed: "任务无法保存。请重启服务恢复后再重试。",
  errBatchPending: "这一项的服务商批处理尚未完成。请先收取结果，再重试或增强。",
  errItemBusy: "这一项仍在转换中。完成后再重试。",
  errNotRetryable: "命令行记录的条目不能在这里重试。请对该文件重新运行 markitai。",
  errNoSource: "这一项没有可重新转换的原始文件或网址。",
  errUploadMissing: "服务器上已没有原始上传文件。请重新上传。",
  errLlmUnavailable: "LLM 处理需要可用的模型。请先在系统设置中添加模型。",
  errProviderBusy: "正在进行的模型请求过多，请稍后再试。",
  errProviderRequest: "服务商设置或环境变量引用无效。",
  errProviderUnavailable: "服务商请求未能完成。请检查接口地址、API key 和网络。",
  errDiscoveryTimeout: "发现模型超时。",
  errSettingsInvalid: "服务不接受这项设置修改。",
  errSettingsMissing: "这项已保存的设置已不存在。请刷新后重试。",
  errSettingsIo: "无法读取或保存配置文件。",
  errConfigMissing: "还没有配置文件。保存一个模型后会自动创建。",
  errSettingsReadOnly: "服务启动时指定了模型或服务商覆盖，设置为只读。请编辑配置文件，并在不带这些参数的情况下重启服务。",
  errStaleRevision: "设置已在其他窗口被修改。",
  errConfigChanged: "保存时配置文件被修改。请重新加载设置后再试。",
  errAmbiguousModel: "有多个模型使用这个路由名。请编辑配置文件使其唯一。",
  errSettingsDurability: "配置已保存，但无法确认已安全写入磁盘。请在下次修改前重新加载设置。",
  errConfigTooLarge: "配置将超过 8 MiB 的上限。",
  errNoModel: "尚未配置模型。请在系统设置中添加模型，或设置 MODEL 和服务商 API key。",
  errOcrUnavailable: "本机 OCR 在此系统上不可用。",
  errOcrVisionSelection: "当前设置选择了 macOS Vision，它需要 macOS 11 或更高版本。请取消 MARKITAI_OCR_BACKEND，或改为 paddle，然后重启服务。",
  errOcrBackendSetting: "请将 MARKITAI_OCR_BACKEND 设为 vision 或 paddle，或取消该变量，然后重启服务。",
  errOcrModelMissing: "缺少本机 OCR 模型。请在运行服务的电脑上联网执行 markitai doctor --fix，或按原始错误中的校验与手动下载说明安装，再重试。",
  errOcrModelCorrupt: "本机 OCR 模型已损坏。请在运行服务的电脑上联网执行 markitai doctor --fix，以校验并替换损坏模型后重试；普通 OCR 不会覆盖它。",
  errOcrModelPath: "本机 OCR 安装路径不安全，或在操作期间发生了变化。请在运行服务的电脑上执行 markitai doctor，检查所指路径的所有权、权限和链接。解决路径问题后再重试；自动修复无法修复不安全路径。",
  errOcrModelDownload: "无法下载本机 OCR 模型。请检查运行服务的电脑的网络和代理，然后执行 markitai doctor --fix，或按原始错误中的校验与手动下载说明安装。",
  errOcrModelPreparation: "本机 OCR 模型准备失败。请在运行服务的电脑上执行 markitai doctor，查看原始错误后再重试。",
  errUnsupported: "这里不支持这项转换。",
  errFetch: "无法抓取此网页。",
  errConversion: "无法转换这个文档。请确认原文件能正常打开，具体原因见错误详情。",
  errInvalidInput: "输入无效。",
  errConfig: "服务配置无法完成这次转换。",
  errIo: "服务无法读写文件。",
  errSourceMissing: "找不到源文件。",
  errEnhanceNoResult: "LLM 增强没有生成增强版本。",
  errNoOutput: "转换没有生成输出。",
  errTimeout: "转换耗时过长，已被停止。",
  errFetchTimeout: "抓取网页超时。",
  errModelFailed: "模型未能完成这个文档。",
  errUnreachable: "无法连接到这个网站。请检查网址以及服务器的网络连接。",
  errFetchPolicy: "服务配置已关闭远程抓取。",
  errNoContent: "网页中没有可提取的内容。",
  errTooLarge: "输入超出支持的大小上限（{limit}）。",
  errPageNotFound: "网页不存在（HTTP {status}）。",
  errPageDenied: "网站拒绝访问（HTTP {status}）。",
  errPageRate: "网站限制了请求频率（HTTP {status}），请稍后再试。",
  errPageServer: "网站服务器出错（HTTP {status}）。",
  errPageHttp: "网站返回 HTTP {status}。",
  errModelTimeout: "模型响应超时。",
  errModelNotFound: "找不到模型或其接口（HTTP {status}）。请检查模型标识和 API base URL。",
  errModelDenied: "服务商拒绝了凭据（HTTP {status}）。请检查 API key。",
  errModelRate: "服务商限制了请求频率（HTTP {status}），请稍后再试。",
  errModelServer: "服务商服务器出错（HTTP {status}）。",
  errModelHttp: "服务商返回 HTTP {status}。",
  errModelRegion: "服务商在你所在的地区不提供这个模型（HTTP {status}）。请换用其它服务商或模型。",
  errModelQuota: "服务商账户的额度已用完或需要开通付费（HTTP {status}）。",
  errModelUnavailable: "服务商不提供这个模型（HTTP {status}）。请检查模型标识。",
  errHistoryNotSaved: "这个任务无法保存到历史。已完成的文件在服务停止前仍可下载。",
  errRollback: "恢复先前结果失败。请重启服务进行恢复。",
  needToken: "没有权限，请用带访问令牌的链接重新打开页面",
  needInput: "请至少选择一个文件或输入一个 URL。",
  nothingToStop: "已经没有等待中的项，剩余的项都在转换中。",
  itemStopped: "在开始转换前已停止 · 可以重试",
  itemShutdown: "服务已停止，转换被取消 · 可以重试",
  unsupportedFormat: "不支持这种文件类型：{format}。",
  probeOk: "{model} 已响应。",
  probeUnreachable: "无法连接到模型接口。",
  probeConfig: "凭据或接口设置无效或缺失。",
  probeUnsupported: "内置运行时不支持这个服务商。",
  probeFailed: "连接测试失败。",
  probeBadResponse: "模型接口返回了无法读取的响应。",
  discoveryNeedsBase: "请先填写 API 地址。",
  discoveryNoRuntime: "这个服务商需要的登录或本地运行时在这里不可用。",
  discoveryFailed: "发现模型失败。请检查接口地址、凭据和服务商状态。",
  discoveryStaleShown: "刷新失败，显示的是之前发现的模型。",
  discoveryAzure: "Azure 列出的是区域基础模型；路由仍需要部署名称。",
  discoveryPaged: "服务商还有更多模型，这里只显示第一页。",
  discoveryOfficial: "由已登录的官方 {name} 运行时报告的模型。",
  discoveryOfficialMissing: "官方 {name} 运行时或其登录不可用。",
};

export const MESSAGES: Record<Locale, Record<MessageKey, string>> = { en, zh };

type Values = Record<string, string | number>;

export function fill(text: string, values?: Values): string {
  if (!values) return text;
  return text.replace(/\{(\w+)\}/g, (match, name: string) => (name in values ? String(values[name]) : match));
}

export function message(locale: Locale, key: MessageKey, values?: Values): string {
  return fill(MESSAGES[locale][key] ?? en[key], values);
}

/** Localized text plus the service's own wording when the text replaced it. */
export interface Described {
  text: string;
  detail: string;
}

const API_REASONS: Record<string, MessageKey> = {
  internal_error: "errInternal",
  shutting_down: "errShuttingDown",
  token_required: "needToken",
  job_id_collision: "errTryAgain",
  host_not_allowed: "errHostNotAllowed",
  origin_not_allowed: "errOriginNotAllowed",
  settings_forbidden: "errSettingsForbidden",
  request_too_large: "errRequestTooLarge",
  form_field_too_large: "errRequestTooLarge",
  file_too_large: "errFileTooLarge",
  too_many_items: "errTooManyItems",
  invalid_url: "errInvalidUrl",
  unsupported_url_scheme: "errUrlScheme",
  remote_url_forbidden: "errRemoteUrl",
  empty_job: "needInput",
  invalid_options: "errInvalidOptions",
  remote_processing_forbidden: "errRemoteProcessingForbidden",
  remote_processing_confirmation_required: "errRemoteConfirmation",
  remote_processing_disabled: "errRemoteDisabled",
  cloudflare_unavailable: "errCloudflareUnavailable",
  unknown_preset: "errInvalidOptions",
  invalid_multipart: "errInvalidRequest",
  invalid_urls: "errInvalidRequest",
  invalid_url_list: "errInvalidUrlList",
  empty_url_list: "errEmptyUrlList",
  url_list_too_large: "errUrlListTooLarge",
  invalid_output_name: "errInvalidOutputName",
  invalid_retry_body: "errInvalidRequest",
  json_required: "errInvalidRequest",
  invalid_json: "errInvalidRequest",
  job_not_found: "errJobNotFound",
  item_not_found: "errItemNotFound",
  job_running: "errJobRunning",
  job_not_running: "nothingToStop",
  nothing_to_stop: "nothingToStop",
  result_unavailable: "errResultUnavailable",
  result_too_large: "errResultTooLarge",
  file_not_found: "errFileNotFound",
  archive_too_large: "errArchiveTooLarge",
  history_empty: "errHistoryEmpty",
  ticket_invalid: "errDownloadTicket",
  invalid_ticket_path: "errInvalidRequest",
  too_many_tickets: "errTooManyTickets",
  output_identity_conflict: "errOutputConflict",
  output_conflict: "errOutputConflict",
  enhancement_failed: "errEnhanceNoResult",
  no_output: "errNoOutput",
  persistence_failed: "errPersistenceFailed",
  batch_pending: "errBatchPending",
  item_busy: "errItemBusy",
  not_retryable: "errNotRetryable",
  no_source: "errNoSource",
  upload_missing: "errUploadMissing",
  llm_unavailable: "errLlmUnavailable",
  provider_busy: "errProviderBusy",
  invalid_provider_request: "errProviderRequest",
  provider_unavailable: "errProviderUnavailable",
  discovery_timeout: "errDiscoveryTimeout",
  invalid_settings_request: "errSettingsInvalid",
  settings_entry_missing: "errSettingsMissing",
  settings_io_failed: "errSettingsIo",
  config_missing: "errConfigMissing",
  settings_read_only: "errSettingsReadOnly",
  stale_revision: "errStaleRevision",
  config_changed: "errConfigChanged",
  ambiguous_legacy_model_name: "errAmbiguousModel",
  settings_durability_unknown: "errSettingsDurability",
  config_too_large: "errConfigTooLarge",
};

const API_CODES: Record<string, MessageKey> = {
  bad_request: "errBadRequest",
  unauthorized: "needToken",
  forbidden: "errForbidden",
  not_found: "errNotFound",
  method_not_allowed: "errBadRequest",
  conflict: "errConflict",
  payload_too_large: "errRequestTooLarge",
  invalid_request: "errInvalidRequest",
  rate_limited: "errBusy",
  unavailable: "errUnavailable",
  server_error: "errInternal",
};

/** Core conversion codes (as in the bindings' envelope) and the service's own item causes. */
const ITEM_CODES: Record<string, MessageKey> = {
  no_model_configured: "errNoModel",
  unsupported: "errUnsupported",
  fetch_error: "errFetch",
  conversion_error: "errConversion",
  invalid_input: "errInvalidInput",
  is_directory: "errInvalidInput",
  invalid_json: "errInvalidInput",
  config_error: "errConfig",
  io_error: "errIo",
  not_found: "errSourceMissing",
  file_not_found: "errSourceMissing",
  upload_missing: "errUploadMissing",
  internal_error: "errInternal",
  enhancement_failed: "errEnhanceNoResult",
  no_output: "errNoOutput",
  output_conflict: "errOutputConflict",
  output_identity_conflict: "errOutputConflict",
};

const HTTP_FAMILY = ["NotFound", "Denied", "Rate", "Server", "Http"] as const;
function httpKey(prefix: "errPage" | "errModel", status: string): MessageKey {
  const code = Number(status);
  const family =
    code === 404 || code === 410
      ? "NotFound"
      : code === 401 || code === 403
        ? "Denied"
        : code === 429
          ? "Rate"
          : code >= 500
            ? "Server"
            : "Http";
  return `${prefix}${family}` as MessageKey;
}

/** The fixed causes the core appends to a model's HTTP refusal. */
const MODEL_REASONS: Record<string, MessageKey> = {
  "the model is not available in this region": "errModelRegion",
  "the account's quota or billing does not allow this request": "errModelQuota",
  "the model is unavailable": "errModelUnavailable",
};

type Shape = [RegExp, (match: RegExpExecArray, kind?: string) => [MessageKey, Values?]];
const modelHttp = (match: RegExpExecArray): [MessageKey, Values] => [
  MODEL_REASONS[match[2] ?? ""] ?? httpKey("errModel", match[1] ?? ""),
  { status: match[1] ?? "" },
];

// Model installation/network failures can be retried after the prerequisite is fixed.
// Share the exact shapes with localization so retryability cannot drift from the message.
const OCR_MODEL_MISSING = /^Local OCR needs the model [\s\S]+, which is not installed: /;
const OCR_MODEL_DOWNLOAD = /^Local OCR download of [\s\S]+ failed \(/;

/** First match wins: specific shapes precede general ones. */
const ITEM_SHAPES: Shape[] = [
  [/^No model configured\b/, () => ["errNoModel"]],
  [
    /^MARKITAI_OCR_BACKEND=vision selects macOS Vision, which needs macOS 11 or later; unset it or choose paddle$/,
    () => ["errOcrVisionSelection"],
  ],
  [/^MARKITAI_OCR_BACKEND must be vision or paddle$/, () => ["errOcrBackendSetting"]],
  [OCR_MODEL_MISSING, () => ["errOcrModelMissing"]],
  [OCR_MODEL_DOWNLOAD, () => ["errOcrModelDownload"]],
  [
    /^Local OCR models: [\s\S]*; run `markitai doctor --fix` to replace this damaged managed model; ordinary OCR will not overwrite it$/,
    () => ["errOcrModelCorrupt"],
  ],
  [
    /^Local OCR models: [\s\S]*(?:installation (?:directory (?:disappeared|changed or is unsafe|is a link, junction or special file|changed while opening)|file (?:disappeared|changed while it was held|is a link, junction or special file|must be private, owned by this user and have one hard link)|lock (?:disappeared|was replaced|changed while opening))|managed home must be owned by this user and not writable by others|managed installation directory must be owned by this user and private|installation paths must not contain parent components|installation path is outside its managed root|unsafe model path)$/,
    () => ["errOcrModelPath"],
  ],
  [/^Local OCR models: /, () => ["errOcrModelPreparation"]],
  [/^Local OCR (?:requires|of multi-page)|^PDF OCR is not implemented/, () => ["errOcrUnavailable"]],
  [/^LLM returned HTTP (\d{3})\b(?:: (.+))?/, modelHttp],
  [/^LLM request timed out/, () => ["errModelTimeout"]],
  [/^LLM request failed$/, () => ["probeUnreachable"]],
  [/^LLM enhancement did not produce/, () => ["errEnhanceNoResult"]],
  [
    /^(?:LLM (?:returned|response|output|document response|per-document|dollar budget)|Cannot read LLM response)/,
    () => ["errModelFailed"],
  ],
  [/^HTTP (\d{3})\b/, (match) => [httpKey("errPage", match[1] ?? ""), { status: match[1] ?? "" }]],
  [/timed out|\btimeout\b/i, (_match, kind) => [kind === "url" ? "errFetchTimeout" : "errTimeout"]],
  [
    /error sending request|dns error|failed to lookup address|connection refused|tcp connect error|Cannot resolve URL hostname|URL has no hostname/i,
    () => ["errUnreachable"],
  ],
  [/^Remote fetching is disabled by policy/, () => ["errFetchPolicy"]],
  [/no extractable content|returned no content|returned empty content/, () => ["errNoContent"]],
  [/^file exceeds upload limit/, () => ["errFileTooLarge"]],
  [/exceeds (?:the )?(\d+(?:\.\d+)? ?[KMG]iB)/, (match) => ["errTooLarge", { limit: match[1] ?? "" }]],
];

/** Provider probe and discovery details, which the core words as fixed phrases. */
const NOTE_SHAPES: Shape[] = [
  [/^Model connection test timed out$/, () => ["errModelTimeout"]],
  [/^Model connection request failed$/, () => ["probeUnreachable"]],
  [/^Model connection returned HTTP (\d{3})(?:: (.+))?$/, modelHttp],
  [/^Model credentials or endpoint configuration are invalid or unavailable$/, () => ["probeConfig"]],
  [/^This model provider is not supported by the native runtime$/, () => ["probeUnsupported"]],
  [/^(?:Model connection test failed|Cannot create model connection client)$/, () => ["probeFailed"]],
  [
    /^(?:Model connection response |Cannot read model connection response|Model connection returned an invalid completion)/,
    () => ["probeBadResponse"],
  ],
  [/^An API endpoint is required$/, () => ["discoveryNeedsBase"]],
  [/^This provider requires an unavailable OAuth or local runtime integration$/, () => ["discoveryNoRuntime"]],
  [/^Too many model discovery requests are active$/, () => ["errProviderBusy"]],
  [/^Model discovery (?:wait )?timed out$/, () => ["errDiscoveryTimeout"]],
  [/^Model discovery (?:did not complete|failed\b)/, () => ["discoveryFailed"]],
  [/^Refresh failed; showing previously discovered models$/, () => ["discoveryStaleShown"]],
  [/^Azure lists regional base models/, () => ["discoveryAzure"]],
  [/^The provider has more models; only its bounded first page is shown$/, () => ["discoveryPaged"]],
  [/^Models reported by the authenticated official (\w+) runtime$/, (match) => ["discoveryOfficial", { name: match[1] ?? "" }]],
  [
    /^Official (\w+) runtime or (?:subscription )?authentication is unavailable$/,
    (match) => ["discoveryOfficialMissing", { name: match[1] ?? "" }],
  ],
  [/^([\s\S]+) responded$/, (match) => ["probeOk", { model: match[1] ?? "" }]],
];

const PERSISTENCE: Record<string, MessageKey> = {
  "history could not be persisted; completed artifacts remain available until shutdown": "errHistoryNotSaved",
  "output rollback failed; restart to recover the previous result": "errRollback",
  "item deletion rollback failed; restart to recover": "errRollback",
};

function shaped(locale: Locale, text: string, shapes: Shape[], kind?: string): Described | null {
  for (const [pattern, pick] of shapes) {
    const match = pattern.exec(text);
    if (match) {
      const [key, values] = pick(match, kind);
      return { text: message(locale, key, values), detail: text };
    }
  }
  return null;
}

/** Every key the tables can produce, for the dictionary tests. */
export function producibleKeys(): MessageKey[] {
  const keys = new Set<MessageKey>([
    ...Object.values(API_REASONS),
    ...Object.values(API_CODES),
    ...Object.values(ITEM_CODES),
    ...Object.values(PERSISTENCE),
    ...Object.values(MODEL_REASONS),
  ]);
  for (const prefix of ["errPage", "errModel"] as const) for (const family of HTTP_FAMILY) keys.add(`${prefix}${family}`);
  const probe = ["", "200"] as unknown as RegExpExecArray;
  for (const shapes of [ITEM_SHAPES, NOTE_SHAPES]) {
    for (const [, pick] of shapes) for (const kind of ["file", "url"]) keys.add(pick(probe, kind)[0]);
  }
  return [...keys];
}

/** An API refusal: by `reason`, then structured `detail.code`, then status `code`. */
export function apiErrorText(locale: Locale, body: unknown): Described | null {
  const value = (body ?? {}) as { detail?: unknown; reason?: unknown; code?: unknown };
  const structured =
    value.detail !== null && typeof value.detail === "object" ? String((value.detail as { code?: unknown }).code ?? "") : "";
  const original = typeof value.detail === "string" ? value.detail : structured;
  const key =
    API_REASONS[typeof value.reason === "string" ? value.reason : ""] ??
    API_REASONS[structured] ??
    API_CODES[typeof value.code === "string" ? value.code : ""];
  if (key) return { text: message(locale, key), detail: original };
  return original ? { text: original, detail: "" } : null;
}

export interface ItemProblem extends Described {
  /** A stop or shutdown: not a fault of the input. */
  hint: boolean;
  /** The folded extension list of an unsupported-format message. */
  formats: string;
}

/** A failed item's error, by code or recognizable shape. */
export function itemErrorText(
  locale: Locale,
  item: { error?: string | null; error_code?: string | null; kind?: string },
): ItemProblem {
  const text = typeof item.error === "string" ? item.error : "";
  const code = item.error_code ?? undefined;
  if (code === "cancelled" || text === "cancelled (stopped by request)") {
    return { text: message(locale, "itemStopped"), detail: text, hint: true, formats: "" };
  }
  if (code === "shutdown" || text === "cancelled (server shutdown)") {
    return { text: message(locale, "itemShutdown"), detail: text, hint: true, formats: "" };
  }
  const split = text.indexOf(" Supported extensions:");
  const head = split > 0 ? text.slice(0, split) : text;
  const formats = split > 0 ? text.slice(split + 1).replace(/^Supported extensions:\s*/, "").replace(/\.$/, "") : "";
  const unsupported = /^Unsupported file format: (.+)\.$/.exec(head);
  if (unsupported) {
    return { text: message(locale, "unsupportedFormat", { format: unsupported[1] ?? "" }), detail: text, hint: false, formats };
  }
  const found =
    shaped(locale, head, ITEM_SHAPES, item.kind) ??
    (code && ITEM_CODES[code] ? { text: message(locale, ITEM_CODES[code]), detail: text } : { text, detail: "" });
  return { ...found, detail: found.detail ? text : "", hint: false, formats };
}

export function serviceNote(locale: Locale, text: unknown): Described {
  if (typeof text !== "string" || !text) return { text: "", detail: "" };
  return shaped(locale, text, NOTE_SHAPES) ?? { text, detail: "" };
}

export function persistenceText(locale: Locale, text: string): Described {
  const key = PERSISTENCE[text];
  return key ? { text: message(locale, key), detail: text } : { text, detail: "" };
}

/** Keep unsupported formats/capabilities blocked; missing models and downloads are recoverable. */
export const isUnsupported = (item: { error?: string | null; error_code?: string | null }): boolean => {
  const error = item.error ?? "";
  if (/^Unsupported file format:/.test(error)) return true;
  return item.error_code === "unsupported" && !OCR_MODEL_MISSING.test(error) && !OCR_MODEL_DOWNLOAD.test(error);
};
