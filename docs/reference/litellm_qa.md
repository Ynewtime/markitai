> Question: litellm 支持 vision_model_list 配置吗

LiteLLM **不支持** `vision_model_list` 这个配置项。

所有模型（包括视觉模型）都统一在 `model_list` 中配置。对于视觉模型，有两种处理方式：

**1. 已知的视觉模型**

LiteLLM 内置识别主流视觉模型，直接配置即可：

```yaml
model_list:
  - model_name: gpt-4-vision-preview
    litellm_params:
      model: openai/gpt-4-vision-preview
      api_key: os.environ/OPENAI_API_KEY
```

**2. 自定义视觉模型**

对于自定义的 OpenAI 兼容模型，可以在 `model_info` 下设置 `supports_vision: True`：

```yaml
model_list:
  - model_name: llava-hf
    litellm_params:
      model: openai/llava-hf/llava-v1.6-vicuna-7b-hf
      api_base: http://localhost:8000
      api_key: fake-key
    model_info:
      supports_vision: True  # 标记支持视觉
```

**检查模型是否支持视觉**

可以用 `litellm.supports_vision()` 函数或调用 `/model_group/info` API：

```python
assert litellm.supports_vision(model="openai/gpt-4-vision-preview") == True
assert litellm.supports_vision(model="openai/gpt-3.5-turbo") == False
```

你是在哪里看到 `vision_model_list` 这个配置的？可能是某个第三方项目的自定义配置。


> Question: model_list 支持 weight 这个参数吗？怎么配置？另外，在哪里可以看到 litellm 全量配置

根据官方文档：

## 1. `weight` 参数配置

**支持**，放在 `litellm_params` 下：

```yaml
model_list:
  - model_name: o1
    litellm_params:
      model: o1
      api_key: os.environ/OPENAI_API_KEY
      weight: 1
  - model_name: o1
    litellm_params:
      model: o1-preview
      api_key: os.environ/OPENAI_API_KEY
      weight: 2  # 这个 deployment 被选中的概率是上面的 2 倍
```

**注意事项**：
- `weight` 只在 `simple-shuffle` 路由策略下生效（这是默认策略）
- 如果不设置 `weight`，也可以用 `rpm`/`tpm` 来做加权负载均衡

---

## 2. 全量配置参考

官方文档提到了两个地方可以查看：

**方式一：启动 Proxy 后访问 Swagger UI**
```
http://<your-proxy-url>/#/config.yaml
例如: http://0.0.0.0:4000/#/config.yaml
```

**方式二：官方文档页面**## 总结

### 1. `weight` 参数配置

**支持**，放在 `litellm_params` 下：

```yaml
model_list:
  - model_name: o1
    litellm_params:
      model: o1
      api_key: os.environ/OPENAI_API_KEY
      weight: 1
  - model_name: o1
    litellm_params:
      model: o1-preview
      api_key: os.environ/OPENAI_API_KEY
      weight: 2  # 被选中概率是上面的 2 倍
```

**注意**：
- `weight` 只在 **`simple-shuffle`** 路由策略下生效（这是默认策略）
- 也可以用 `rpm`/`tpm` 做加权（按配额比例分配）

---

### 2. 全量配置参考

**官方文档页面**：https://docs.litellm.ai/docs/proxy/config_settings

这个页面包含了：
- `model_list` 结构
- `litellm_settings` 所有选项
- `general_settings` 所有选项  
- `router_settings` 所有选项
- 所有环境变量

**Swagger UI**（启动 Proxy 后访问）：
```
http://<your-proxy-url>/#/config.yaml
```

---

### 快速参考：`model_list` 常用字段

```yaml
model_list:
  - model_name: gpt-4           # 用户请求时使用的名称
    litellm_params:             # 传给 litellm.completion() 的参数
      model: azure/gpt-4        # 实际模型
      api_key: os.environ/KEY
      api_base: https://xxx
      rpm: 100                  # 速率限制
      tpm: 100000
      weight: 2                 # 权重
      order: 1                  # 优先级（需 enable_pre_call_checks: true）
      max_parallel_requests: 10 # 最大并发
    model_info:                 # 模型元信息
      supports_vision: true
      base_model: azure/gpt-4-1106-preview  # 用于成本计算
      max_tokens: 128000
```

> Question: #litellm 请问 litellm 的 weight 配置有什么限制吗？比如不同模型之间的 weight 加和需要等于 10？weight 跟 order 参数有什么区别，更推荐哪一个呢？要深入调研最新信息哦

我来帮你深入调研 LiteLLM 的 weight 和 order 配置参数。根据我的深入调研，以下是关于 LiteLLM `weight` 和 `order` 配置的详细解答：

## Weight 配置的限制

**没有硬性加和要求**。Weight 值不需要等于 10 或任何特定总和。LiteLLM 使用的是**相对比例**而非绝对值。

例如以下配置中，`weight: 9` 和 `weight: 1` 意味着第一个部署会被选中约 90% 的次数：

```yaml
model_list:
  - model_name: o1
    litellm_params:
      model: o1
      api_key: os.environ/OPENAI_API_KEY
      weight: 9    # 被选中 90% 的时间
  - model_name: o1
    litellm_params:
      model: o1-preview
      api_key: os.environ/OPENAI_API_KEY
      weight: 1    # 被选中 10% 的时间
```

**Weight 的适用范围**：Weight 仅在 `simple-shuffle` 路由策略（默认策略）下生效。实际上，有社区讨论指出文档曾声称 weight 适用于所有路由策略，但实际上并非如此。

---

## Weight vs Order 的核心区别

| 特性 | Weight | Order |
|------|--------|-------|
| 作用机制 | **概率性**负载均衡 | **确定性**优先级排序 |
| 流量分配 | 按比例分配到所有可用部署 | 始终优先使用最高优先级部署 |
| 回退触发 | 不涉及回退 | 仅当高优先级部署不可用时回退 |
| 依赖条件 | 无特殊要求 | 需要 `enable_pre_call_checks=True` |

### Order 的工作方式

Order 数值越低优先级越高（order=1 > order=2）。当多个部署具有相同 order 值时，路由策略会在它们之间选择。

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-primary
      api_key: os.environ/AZURE_API_KEY
      order: 1  # 最高优先级，始终首选
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-fallback
      api_key: os.environ/AZURE_API_KEY_2
      order: 2  # 仅当 order=1 不可用时使用

router_settings:
  enable_pre_call_checks: true  # 必须开启！
```

---

## 选择建议

**推荐使用 Weight** 的场景：
- 需要在多个部署间**分散负载**
- 想要**渐进式流量迁移**（如新模型测试时先给 10% 流量）
- 希望所有部署都能持续接收请求

**推荐使用 Order** 的场景：
- 有明确的**主备关系**
- 希望**降低成本**（优先用便宜的部署，贵的作为后备）
- 需要**确定性**的故障转移行为

---

## 已知问题

最近有用户报告 `order` 参数可能未按预期工作——即使 order=1 的部署完全健康，order=2 的部署也会收到约 50% 的流量。这个 issue 在 2024 年 12 月底被提出，建议使用时注意测试验证。

如果需要最稳定可靠的负载均衡，LiteLLM 官方推荐生产环境使用 `simple-shuffle`（默认）策略配合 `rpm`/`tpm` 限制。


> Question: litellm 的配置中，model_list.model_name 跟 model_list.model_info.id 有什么区别？如果要做负载均衡，应该怎么配置

根据我的调研，以下是 `model_name` 和 `model_info.id` 的详细区别以及负载均衡配置方法：

## model_name vs model_info.id 的区别

| 字段 | 作用 | 特点 |
|------|------|------|
| `model_name` | **模型组名称** (Model Group) | 用户调用时使用的名称，相同 model_name 的部署会被自动负载均衡 |
| `model_info.id` | **部署唯一标识符** (Deployment ID) | 用于追踪具体是哪个部署处理了请求，调试和监控用 |

### model_name 的作用

模型部署通过 config.yaml 中的 `model_name` 字段被分配到一个"模型组"。当你给 key/user/team 授权时，你授权的是对"模型组"的访问权限。

在 config.yaml 中，`model_name` 参数是面向用户的部署名称。

### model_info.id 的作用

如果不手动指定 `model_info.id`，LiteLLM 会自动生成一个基于 `litellm_params` 哈希的唯一 ID（如 `e889baacd17f591cce4c63639275ba5e8dc60765d6c553e6ee5a504b19e50ddc`）。

在响应中可以通过 `response._hidden_params["model_id"]` 查看实际使用了哪个部署。

---

## 负载均衡配置方法

### 核心原则：相同 model_name = 自动负载均衡

你可以将多个模型部署添加到同一个"模型组"。这样可以最大化你在多个部署之间的速率限制。

### 示例配置

```yaml
model_list:
  # ========== 负载均衡组: gpt-4 ==========
  - model_name: gpt-4                    # 相同的 model_name
    litellm_params:
      model: azure/gpt-4-deployment-1
      api_base: https://us-east.openai.azure.com/
      api_key: os.environ/AZURE_KEY_1
      rpm: 60                            # 每分钟请求数限制
    model_info:
      id: azure-us-east                  # 可选：自定义 ID，便于调试

  - model_name: gpt-4                    # 相同的 model_name → 自动负载均衡
    litellm_params:
      model: azure/gpt-4-deployment-2
      api_base: https://us-west.openai.azure.com/
      api_key: os.environ/AZURE_KEY_2
      rpm: 60
    model_info:
      id: azure-us-west

  - model_name: gpt-4                    # 同组，使用 OpenAI 作为后备
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
      rpm: 100
    model_info:
      id: openai-direct

  # ========== 另一个模型组: claude ==========
  - model_name: claude                   # 不同的 model_name = 独立的组
    litellm_params:
      model: anthropic/claude-3-sonnet
      api_key: os.environ/ANTHROPIC_KEY
```

### 用户调用方式

```bash
# 调用 gpt-4 组 → LiteLLM 自动在 3 个部署间负载均衡
curl -X POST 'http://localhost:4000/chat/completions' \
  -H 'Authorization: Bearer sk-xxx' \
  -d '{"model": "gpt-4", "messages": [...]}'

# 调用 claude 组
curl -X POST 'http://localhost:4000/chat/completions' \
  -d '{"model": "claude", "messages": [...]}'
```

---

## 完整负载均衡配置示例（带 weight/order）

```yaml
model_list:
  # 主力部署 (高权重)
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-primary
      api_key: os.environ/AZURE_KEY
      api_base: https://primary.openai.azure.com/
      weight: 8                          # 80% 流量
      # order: 1                         # 或用 order 做主备模式
    model_info:
      id: primary

  # 备用部署 (低权重)
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-backup
      api_key: os.environ/AZURE_KEY_2
      api_base: https://backup.openai.azure.com/
      weight: 2                          # 20% 流量
      # order: 2
    model_info:
      id: backup

router_settings:
  routing_strategy: simple-shuffle       # 默认策略，weight 仅在此策略下生效
  # enable_pre_call_checks: true         # 如果使用 order，需要开启这个
  num_retries: 3
  allowed_fails: 3
  cooldown_time: 30
```

---

## 总结

| 配置项 | 用途 |
|--------|------|
| `model_name` | **定义负载均衡组**。相同值 = 同一组，自动负载均衡 |
| `model_info.id` | **标识具体部署**。用于日志追踪、调试、监控哪个部署被选中 |
| `litellm_params.model` | **实际调用的模型**。发送给 LLM 提供商的真实模型名 |
| `weight` / `rpm` / `tpm` | **控制负载分配比例** |
| `order` | **控制优先级顺序**（主备模式） |