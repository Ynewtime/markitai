# ROADMAP

## 任务批次 2026010902

### 新特性

1. 特性一：新增 `markit provider select` 命令行功能，支持用户从已配置的 provider 中选取模型直接写入 markit.toml 配置。细分功能点：
   1. 需要优化当前的配置文件设计，区分 Provider 和 Model，`provider test` 和 `provider models` 命令应该针对 Provider 相关配置进行
   2. 对于 select/models 命令，要支持获取模型的 capabilities（如是否支持 text/vision/推理 等）

---

# 归档

## 任务批次 2026010901

### 新特性

1. 特性一：任务级别的日志记录，每次任务生成对应时间戳的日志，同时在日志头部打印当此任务的详细配置，执行完成或者被打断时在该次任务日志尾部记录最终报告
2. 特性二：大模型资源池，支持配置/使用多个 Provider/Models，形成一个可用的负载均衡资源池，从而大大提高 llm 相关任务的并发数支持

### 可靠性

1. 根据 git 未提交的最新修改，进一步提高单元测试覆盖率
2. 审视 ruff 和单元测试，修复相关报错

### 进展

已完成