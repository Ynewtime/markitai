import { useRef } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { useModal } from "../hooks/use-modal.ts";
import type { CloudflareReason, CloudflareScope } from "../lib/cloudflare.ts";
import { Icon } from "./icons.tsx";

export function CloudflareDialog({ t, scope, reason, retry, requestCount, onClose, onConfirm }: {
  t: Dict; scope: CloudflareScope; reason: CloudflareReason | null; retry: boolean; requestCount: number;
  onClose: () => void; onConfirm: () => void;
}) {
  const dialog = useRef<HTMLDivElement>(null);
  const maySend = scope.urls > 0 || scope.candidates > 0;
  useModal(dialog, onClose);
  return <div class="veil" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <div ref={dialog} class="dialog" role="dialog" aria-modal="true" aria-labelledby="cloudflare-title" aria-describedby="cloudflare-scope" tabIndex={-1}>
      <div class="dialog-head">
        <h2 id="cloudflare-title" class="dialog-title">{reason ? t.cloudflareUnavailable : requestCount > 1 ? t.cloudflareBatchTitle(requestCount) : maySend ? t.cloudflareTitle : t.cloudflareLocalTitle}</h2>
        <button type="button" class="icon-btn" aria-label={t.close} onClick={onClose}><Icon name="X" size={16} /></button>
      </div>
      <div class="dialog-body">
        <p id="cloudflare-scope" class="dialog-text">{reason ? t.cloudflareReason(reason) : t.cloudflareScopeCounts(scope.urls, scope.candidates, scope.files, scope.native)}</p>
        {reason ? reason !== "incompatible_strategy" && <p class="dialog-dim">{t.cloudflareSetup}</p> : <>
          {scope.candidates > 0 ? <p class="dialog-dim">{t.cloudflareScopeHint}</p> : scope.native > 0 && <p class="dialog-dim">{t.cloudflareNativeHint}</p>}
          {retry && maySend && <p class="dialog-text">{t.cloudflareRetry}</p>}
          {maySend && <p class="dialog-text">{t.cloudflareCharges}</p>}
          <p class="dialog-dim">{t.cloudflareOnce}</p>
        </>}
        <div class="form-actions">
          <button type="button" class="btn btn-ghost" onClick={onClose}>{t.cancel}</button>
          {!reason && <button type="button" class="btn btn-primary" onClick={onConfirm}>{maySend ? t.cloudflareConfirm : t.cloudflareContinue}</button>}
        </div>
      </div>
    </div>
  </div>;
}
