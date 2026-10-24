// A subscription runtime (Claude Code, Copilot, Codex) that lists no models says
// why: not installed, not a release the adapter speaks, not signed in, or
// failed; with the command to run on the machine the server runs on, one click
// to copy it, and the markitai.dev setup guide at the section for the cause.
// The server's sentence is markitai's own (no runtime output is passed on), so
// it is never shown as the provider's message.
import { useState } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { copyText } from "../lib/clipboard.ts";
import type { Locale } from "../i18n/locale.ts";
import { runtimeGuide } from "../lib/guides.ts";
import type { RuntimeFailure } from "../lib/model-test.ts";
import { Icon } from "./icons.tsx";

/** Claude Code signed in, but not with a subscription (an API key, a cloud): the server says so in its own words. */
const NOT_SUBSCRIPTION = /non-subscription/i;

export function RuntimeNotice({ t, failure, provider, locale }: { t: Dict; failure: RuntimeFailure; provider: string; locale: Locale }) {
  const [copied, setCopied] = useState(false);
  const { name } = failure;
  const title =
    failure.reason === "runtime_missing"
      ? t.runtimeMissingTitle(name)
      : failure.reason === "runtime_unsupported"
        ? t.runtimeUnsupportedTitle(name)
        : failure.reason === "not_signed_in"
          ? t.notSignedInTitle(name)
          : t.runtimeFailedTitle(name);
  const text =
    failure.reason === "runtime_missing"
      ? t.runtimeMissingText
      : failure.reason === "runtime_unsupported"
        ? t.runtimeUnsupportedText(failure.installed, failure.required)
        : failure.reason === "not_signed_in"
          ? NOT_SUBSCRIPTION.test(failure.detail)
            ? t.notSubscriptionText
            : t.notSignedInText
          : t.runtimeFailedText;
  const guide = runtimeGuide(locale, provider, failure.reason);
  return (
    <div class="test-failure runtime-notice" role="alert">
      <Icon name="WarningCircle" size={16} />
      <div class="test-failure-copy">
        <p class="test-failure-title">
          <strong>{title}</strong>
        </p>
        <p>{text}</p>
        {failure.step && (
          <div class="runtime-step">
            <span class="runtime-step-kind">{failure.step.kind === "install" ? t.runtimeStepInstall : t.runtimeStepSignIn}</span>
            <code>{failure.step.command}</code>
            <button
              type="button"
              class="icon-btn"
              aria-label={t.copyCommand}
              title={t.copyCommand}
              onClick={() =>
                void copyText(failure.step?.command ?? "").then((ok) => {
                  setCopied(ok);
                  if (ok) setTimeout(() => setCopied(false), 1500);
                })
              }
            >
              <Icon name={copied ? "Check" : "Copy"} size={14} />
            </button>
          </div>
        )}
        <p class="runtime-links">
          <a class="text-link" href={guide} target="_blank" rel="noopener noreferrer">
            {t.runtimeSetupGuide}
            <span class="sr-only"> {t.opensNewTab}</span>
          </a>
        </p>
      </div>
    </div>
  );
}
