/* The native apps' private core (crates/markitai-app, desktop/DESIGN.md
 * section 3). Not a public binding: it changes with the apps. */
#ifndef MARKITAI_APP_H
#define MARKITAI_APP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MARKITAI_APP_ABI_VERSION 1u

/* Owns len bytes of UTF-8 JSON, without a trailing NUL. Release it exactly once
   with markitai_app_buffer_free. Do not copy this ownership handle. */
typedef struct MarkitaiAppBuffer {
    uint8_t *data;
    size_t len;
} MarkitaiAppBuffer;

uint32_t markitai_app_abi_version(void);

/* The markitai version, such as "2.3.2". Static UTF-8, never free or modify. */
const char *markitai_app_version(void);

/* Each request is UTF-8 JSON, borrowed until return, at most 64 MiB. Each
   response is {"ok":true,"result":{...}} or
   {"ok":false,"error":{"code":"...","message":"..."}}.
   prepare_environment sets environment variables: call it first in main,
   before any other thread starts. convert_json blocks until the conversion
   ends: call it off the UI thread. */
MarkitaiAppBuffer markitai_app_prepare_environment(const uint8_t *request, size_t len);
MarkitaiAppBuffer markitai_app_convert_json(const uint8_t *request, size_t len);
MarkitaiAppBuffer markitai_app_preview_html(const uint8_t *request, size_t len);
MarkitaiAppBuffer markitai_app_claim_output_folder(const uint8_t *request, size_t len);
MarkitaiAppBuffer markitai_app_config_file(void);

/* Clears the struct. Null, or a struct already cleared, is harmless. */
void markitai_app_buffer_free(MarkitaiAppBuffer *buffer);

#ifdef __cplusplus
}
#endif
#endif
