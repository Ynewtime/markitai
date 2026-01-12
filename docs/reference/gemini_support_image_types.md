我来为您深度调研 Gemini 3 Pro 支持的图片输入格式。让我先搜索相关信息。根据我对 Google 官方文档和相关资源的深度调研，以下是 **Gemini 3 Pro 支持的图片输入格式**的完整信息：

---

## 一、支持的图片格式（MIME Types）

Gemini 支持以下图片格式的 MIME 类型：

| 格式 | MIME Type |
|------|-----------|
| **PNG** | `image/png` |
| **JPEG** | `image/jpeg` |
| **WebP** | `image/webp` |
| **HEIC** | `image/heic` |
| **HEIF** | `image/heif` |

---

## 二、图片输入方式

### 1. 内联数据（Inline Data）
- 适用于较小文件
- 总请求大小限制为 **20 MB**（包括文本提示、系统指令和所有内联文件）
- 图片需要 Base64 编码

### 2. Files API 上传
- 适用于较大文件或需要重复使用的图片
- 通过 Files API 上传的文件保留 48 小时
- 支持更大的文件尺寸

---

## 三、图片尺寸与 Token 计算

### 分辨率处理
图片像素数量没有特定限制。较大的图片会被缩小并填充到最大分辨率 3072x3072，同时保持原始宽高比。

### Token 计算规则
对于 Gemini 2.0 及以上版本，两个维度都 ≤384 像素的图片计为 258 个 token。较大的图片会被分割成 768x768 像素的图块，每个图块消耗 258 个 token。

### Gemini 3 的 media_resolution 参数
media_resolution 参数根据输入类型映射到不同的 token 数量。图片线性缩放：
- `media_resolution_low`: 280 tokens
- `media_resolution_medium`: 560 tokens  
- `media_resolution_high`: 1120 tokens

---

## 四、图片数量限制

| 模型版本 | 每请求最大图片数 |
|----------|------------------|
| Gemini 2.5 Pro/Flash, 2.0 Flash, 1.5 Pro, 1.5 Flash | 3,600 个图片文件 |
| Gemini 3 Pro Image（用于图片生成） | 最多 14 张参考图片 |

---

## 五、Gemini 3 Pro Image（图片生成变体）的特殊规格

对于 Gemini 3 Pro Image（也称为 Nano Banana Pro）：
- **输入模态**：支持文本和图片（不接受音频或视频作为图片生成输入）
- **每提示最大图片数**：14 张
- **最大图片上传大小**：每张输入图片 7 MB
- **支持的宽高比**：1:1, 3:2, 16:9, 9:16, 21:9 等
- **输出分辨率**：支持 4K/4096px

---

## 六、最佳实践建议

1. 确保图片正确旋转
2. 使用清晰、不模糊的图片
3. 当使用单张图片配合文本时，将文本提示放在 contents 数组中图片部分的后面
4. 为确保广泛兼容性，优先使用 PNG/JPEG/WebP 格式
5. 对于照片，推荐使用高质量 JPEG（质量 90-95）或高质量有损 WebP（q ≈ 80-90）

---

## 七、上下文窗口

Gemini 3 模型支持 **100 万 token** 的输入上下文窗口和最高 **64k token** 的输出。

---

如需更多技术细节，可参考 Google 官方文档：
- [Image Understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
- [Media Resolution](https://ai.google.dev/gemini-api/docs/media-resolution)
- [Gemini 3 Developer Guide](https://ai.google.dev/gemini-api/docs/gemini-3)