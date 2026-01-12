# E2E Tests: End-to-End Testing

端到端测试目录。目前该目录为空，因为原有的弹性特性测试已移至单元测试。

---

## 历史变更

以下测试已迁移至 `tests/unit/`：

| 原文件 | 新位置 | 说明 |
|--------|--------|------|
| `test_chaos_provider.py` | `tests/unit/test_chaos_provider.py` | ChaosMockProvider 测试 |
| `test_aimd_limiter.py` | `tests/unit/test_aimd_limiter.py` | AIMD 限速器测试 |
| `test_bounded_queue.py` | `tests/unit/test_bounded_queue.py` | 背压队列测试 |
| `test_dead_letter_queue.py` | `tests/unit/test_dead_letter_queue.py` | 死信队列测试 |

这些测试不需要外部服务，可以通过模拟完成，因此属于单元测试范畴。

---

## 真正的 E2E 测试

需要外部服务（如真实 LLM API）的测试应放在此目录。例如：

- 需要 API Key 的 LLM 调用测试
- 与真实服务集成的测试

### 运行需要 API Key 的测试

```bash
# 设置 API Keys
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export GOOGLE_API_KEY=your_key

# 运行 e2e 测试
pytest -m e2e
```

---

## CLI 端到端验证

使用 CLI 命令验证实际功能：

```bash
# Provider 连接测试
markit provider test

# 单文件转换
markit convert document.pdf --llm

# 批量转换
markit batch ./docs -o ./output -r
```
