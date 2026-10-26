// Notice titles by stable code (contract v1.26 `notices[].code`), for the warnings no wording
// pattern in lib/notices.ts recognizes. A code names a family of sentences, so a title says what
// the family means; the warning's own words stay behind Details. Codes are only ever added: a code
// not listed here is shown in the server's words.
import type { Locale } from "./locale.ts";

type Title = (count: number) => string;
const n = (count: number, one: string, many: string): string => (count === 1 ? one : many.replace("{n}", String(count)));

const TITLES: Record<string, Record<Locale, Title>> = {
  "pdf.page": {
    en: (c) => n(c, "A note about how a PDF page's text was read", "{n} notes about how PDF pages' text was read"),
    zh: () => "关于 PDF 页面文字读取方式的说明",
    ja: () => "PDF ページの文字の読み取りについての注意",
  },
  "pdf.page_without_text": {
    en: (c) => n(c, "PDF pages without readable text: turn on OCR and convert again", "{n} notes about PDF pages without readable text: turn on OCR and convert again"),
    zh: () => "PDF 有页面没有可读文字：打开 OCR 后重新转换",
    ja: () => "PDF に読み取れる文字のないページがあります：OCR をオンにして変換し直してください",
  },
  "pdf.page_plain_text": {
    en: (c) => n(c, "A PDF page was read as plain text: its reading order and styling may differ", "{n} PDF pages were read as plain text: their reading order and styling may differ"),
    zh: () => "有 PDF 页面按纯文本读取：阅读顺序和文字样式可能与原文不同",
    ja: () => "PDF ページをプレーンテキストとして読み取りました：読む順序や書式が異なる場合があります",
  },
  "pdf.embedded_image": {
    en: (c) => n(c, "OCR could not read an image in the PDF; the image is kept", "OCR could not read {n} images in the PDF; the images are kept"),
    zh: () => "OCR 无法读取 PDF 中的图片；图片已保留",
    ja: () => "PDF 内の画像を OCR で読み取れませんでした。画像はそのまま残しています",
  },
  "pdf.layout": {
    en: () => "PDF layout refinement was skipped; the page reader's output is kept",
    zh: () => "已跳过 PDF 版面优化；保留页面读取器的原始输出",
    ja: () => "PDF のレイアウト調整を省略しました。ページ読み取りの出力をそのまま使っています",
  },
  "pdf.hidden_text": {
    en: () => "Hidden text in the PDF was left out: it may be a prompt injection",
    zh: () => "已略去 PDF 中的隐藏文字：它可能是提示注入",
    ja: () => "PDF の非表示テキストを除外しました：プロンプトインジェクションの可能性があります",
  },
  "ocr.no_text": {
    en: (c) => n(c, "OCR found no readable text", "OCR found no readable text in {n} places"),
    zh: () => "OCR 没有识别到可读文字",
    ja: () => "OCR で読み取れる文字が見つかりませんでした",
  },
  "ocr.capped": {
    en: () => "OCR read only part of a very dense image, from the top",
    zh: () => "图片文字过多，OCR 只从顶部读取了一部分",
    ja: () => "文字の多い画像のため、OCR は上から一部だけを読み取りました",
  },
  "ocr.unread": {
    en: (c) => n(c, "OCR could not read part of the document", "OCR could not read {n} parts of the document"),
    zh: () => "OCR 无法读取文档的部分内容",
    ja: () => "OCR で文書の一部を読み取れませんでした",
  },
  "ocr.unavailable": {
    en: () => "Local OCR had a problem; see the details",
    zh: () => "本地 OCR 出现问题；详见详情",
    ja: () => "ローカル OCR で問題が発生しました。詳細をご覧ください",
  },
  "llm.enhancement_failed": {
    en: (c) => n(c, "LLM enhancement failed; the result without it is kept", "LLM enhancement failed for {n} documents; their results without it are kept"),
    zh: () => "LLM 增强失败；保留未增强的结果",
    ja: () => "LLM による補強に失敗しました。補強前の結果を残しています",
  },
  "llm.cache_unavailable": {
    en: () => "The LLM cache could not be used; answers were requested without it",
    zh: () => "无法使用 LLM 缓存；已在不使用缓存的情况下请求回答",
    ja: () => "LLM キャッシュを使えなかったため、キャッシュなしで回答を求めました",
  },
  "llm.cache_entry_ignored": {
    en: () => "A damaged LLM cache entry was ignored",
    zh: () => "已忽略一条损坏的 LLM 缓存",
    ja: () => "壊れた LLM キャッシュ項目を無視しました",
  },
  "llm.text_fallback": {
    en: () => "Reading the rendered pages with the model failed; their text was used instead",
    zh: () => "模型读取页面截图失败；改用页面文字",
    ja: () => "モデルによるページ画像の読み取りに失敗したため、ページの文字を使いました",
  },
  "text_only.no_text": {
    en: () => "No text was recognized, so the text-only result is empty",
    zh: () => "没有识别到文字，纯文本结果为空",
    ja: () => "文字を認識できなかったため、テキストのみの結果は空です",
  },
  "image.analysis_failed": {
    en: (c) => n(c, "An image analysis failed; the image and the Markdown are kept", "{n} image analyses failed; the images and the Markdown are kept"),
    zh: () => "图片分析失败；图片和 Markdown 均已保留",
    ja: () => "画像の分析に失敗しました。画像と Markdown は残しています",
  },
  "image.not_localized": {
    en: (c) => n(c, "An image could not be saved locally; its original address is kept", "{n} images could not be saved locally; their original addresses are kept"),
    zh: () => "有图片无法保存到本地；保留其原始地址",
    ja: () => "画像をローカルに保存できなかったため、元のアドレスを残しています",
  },
  "image.webp_as_jpeg": {
    en: () => "Pictures were written as JPEG: this build has no lossy WebP encoder",
    zh: () => "图片已保存为 JPEG：此版本没有有损 WebP 编码器",
    ja: () => "このビルドには非可逆 WebP エンコーダーがないため、画像を JPEG で保存しました",
  },
  "office.renderer_missing": {
    en: () => "Page screenshots and OCR of Office pages need LibreOffice; the document's own text was converted",
    zh: () => "Office 页面截图和 OCR 需要 LibreOffice；已转换文档自身的文字",
    ja: () => "Office ページのスクリーンショットと OCR には LibreOffice が必要です。文書の文字は変換しました",
  },
  "office.page": {
    en: (c) => n(c, "A note about an Office page's rendering", "{n} notes about Office pages' rendering"),
    zh: () => "关于 Office 页面渲染的说明",
    ja: () => "Office ページの描画についての注意",
  },
  "office.screenshot_format": {
    en: () => "Office screenshots use lossless WebP",
    zh: () => "Office 截图使用无损 WebP",
    ja: () => "Office のスクリーンショットは可逆 WebP です",
  },
  "office.count_verified": {
    en: () => "A count was checked against LibreOffice",
    zh: () => "已用 LibreOffice 核对数量",
    ja: () => "数を LibreOffice で確認しました",
  },
  "spreadsheet.screenshot_scope": {
    en: () => "A note about what workbook screenshots cover",
    zh: () => "关于工作簿截图范围的说明",
    ja: () => "ブックのスクリーンショットの範囲についての注意",
  },
  "spreadsheet.font_color": {
    en: () => "A note about workbook fonts without a declared color",
    zh: () => "关于未声明颜色的工作簿字体的说明",
    ja: () => "色の指定がないブックのフォントについての注意",
  },
  "spreadsheet.rendering": {
    en: () => "A note about how a workbook's sheets were laid out",
    zh: () => "关于工作簿排版的说明",
    ja: () => "ブックのシートのレイアウトについての注意",
  },
  "spreadsheet.numbers_formatting": {
    en: () => "Numbers display formats are simplified",
    zh: () => "Numbers 的显示格式已简化",
    ja: () => "Numbers の表示形式を簡略化しました",
  },
  "spreadsheet.numbers_objects": {
    en: () => "Numbers charts, shapes, text boxes and images are not extracted",
    zh: () => "未提取 Numbers 中的图表、形状、文本框和图片",
    ja: () => "Numbers のグラフ、図形、テキストボックス、画像は抽出していません",
  },
  "table.merged_cells": {
    en: () => "Merged table cells keep their text in the first cell; the others are empty",
    zh: () => "合并单元格的文字保留在第一个单元格，其余为空",
    ja: () => "結合セルの文字は先頭のセルに入れ、残りは空にしています",
  },
  "table.malformed": {
    en: (c) => n(c, "A table in the source is malformed", "{n} tables in the source are malformed"),
    zh: () => "源文件中有格式不正确的表格",
    ja: () => "元の文書に形式の崩れた表があります",
  },
  "browser.screenshot_failed": {
    en: () => "The page screenshot failed; the page's text is kept",
    zh: () => "网页截图失败；已保留页面文字",
    ja: () => "ページのスクリーンショットに失敗しました。ページの文字は残しています",
  },
  "cloudflare.not_used": {
    en: (c) => n(c, "Cloudflare was not used for a file: OCR and screenshots need the local reader", "Cloudflare was not used for {n} files: OCR and screenshots need the local reader"),
    zh: () => "有文件未使用 Cloudflare：OCR 和截图需要本地读取",
    ja: () => "OCR とスクリーンショットにはローカルの読み取りが必要なため、Cloudflare を使わなかったファイルがあります",
  },
  "mail.eml": {
    en: () => "A note about the email message",
    zh: () => "关于邮件的说明",
    ja: () => "メールについての注意",
  },
  "mail.msg": {
    en: () => "A note about the Outlook message",
    zh: () => "关于 Outlook 邮件的说明",
    ja: () => "Outlook メールについての注意",
  },
};

/** Whether the workbench has its own title for `code`. */
export const hasCodeTitle = (code: string): boolean => Object.hasOwn(TITLES, code);

/** The title of a notice group of `code`, or null for a code without one. */
export function codeTitle(locale: Locale, code: string, count: number): string | null {
  return TITLES[code]?.[locale](count) ?? null;
}
