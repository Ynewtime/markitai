# Integration Tests

集成测试，使用真实测试文件验证文档转换功能。

---

## 测试文件

| 文件 | 功能 | 依赖 |
|------|------|------|
| `test_batch_conversion.py` | 批量文档转换 | 测试文件 (`tests/fixtures/documents/`) |
| `test_resilience.py` | 弹性场景测试 | 无外部依赖 |
| `test_logging.py` | 任务日志测试 | 无外部依赖 |

---

## 运行方式

```bash
# 运行所有集成测试
pytest tests/integration/ -v

# 运行单个测试文件
pytest tests/integration/test_batch_conversion.py -v

# 跳过需要 LibreOffice 的测试
pytest tests/integration/ -v -k "not legacy"
```

---

## 测试内容

### 1. 批量转换测试 (`test_batch_conversion.py`)

验证真实文档转换功能：

- **输入目录验证**：检查测试文件是否存在
- **输出命名**：验证 `<name>.<ext>.md` 格式
- **格式转换**：测试 docx, xlsx, pptx, pdf 等格式
- **旧格式支持**：测试 doc, ppt, xls（需要 LibreOffice）
- **图像提取**：验证 PDF/PPTX 图片提取
- **清理验证**：确保不留下临时文件

### 2. 弹性测试 (`test_resilience.py`)

验证系统在高负载和故障场景下的稳定性：

**Scenario 1 - Marathon（马拉松）**
- 处理 100 个文件（CI 优化，原 SPEC 为 1000）
- 使用 ChaosMockProvider 模拟故障
- 验证无崩溃，所有文件最终完成

**Scenario 2 - Interrupter（中断恢复）**
- 验证 state.json 完整性
- 测试中断后恢复不重复处理
- DLQ 防止无限重试

**Scenario 3 - Backpressure（背压）**
- BoundedQueue 背压测试
- DLQ 元数据追踪测试

### 3. 日志测试 (`test_logging.py`)

验证任务日志功能：

- 日志目录创建（`.logs/`）
- 日志文件命名（`convert_*.log`）
- 日志内容格式（Task Configuration、Task Completed）

---

## 测试夹具

测试文件位于 `tests/fixtures/documents/`：

```
tests/fixtures/documents/
├── file-example_PDF_500_kB.pdf
├── file-sample_100kB.doc
├── file_example_PPT_250kB.ppt
├── file_example_XLS_100.xls
├── file_example_XLSX_100.xlsx
└── Free_Test_Data_500KB_PPTX.pptx
```

---

## 依赖项

- **LibreOffice**：旧格式（.doc, .ppt, .xls）转换需要
- **测试文件**：位于 `tests/fixtures/documents/`

```bash
# 检查 LibreOffice 是否安装
which soffice || which libreoffice

# 如未安装，旧格式测试会自动跳过
```

---

## 与其他测试的关系

| 测试类型 | 目录 | 特点 |
|----------|------|------|
| Unit | `tests/unit/` | 快速，隔离测试单个模块 |
| **Integration** | `tests/integration/` | 使用真实文件，验证模块集成 |
| E2E | `tests/e2e/` | 验证弹性机制，模拟极端场景 |
