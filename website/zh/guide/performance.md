# 转换性能

URL 提取本地优先：短文和中日韩正文通常在 static 策略内就能完成，不必启动
浏览器。本地文件转换只加载该格式需要的后端：普通 Office 文档走专用路径，
复杂内容、公式和含歧义的表格值保留回退读取器。完整的策略级联见
[抓取策略](./fetch-policy)。

## 实测结果（2026-09-14）

在 macOS、Python 3.12.14 环境下，相对冻结的优化前源码快照，综合提速
**6.01×**：

| 等权负载 | 提速 |
| --- | ---: |
| 通过本地回环 HTTP 获取的五个 URL 页面 | 5.83× |
| DOCX、XLSX、PPTX、PDF 的冷 CLI 转换 | 5.61× |
| 30 个 TXT/HTML/DOCX 文件的批处理 | 6.64× |

综合值是三组的几何平均，覆盖 49 个固定样例、每个样例三个新进程，并关闭模型
增强、远程抓取和缓存。该次审计也比对了产物：40 项转换器/API 快照、23 个
复杂表格样例和 208 个网页样例的输出完全一致。

这是冷启动 CLI 的聚合结果，别拿它当某个具体文档或机器的保证值。URL 计时走本地
回环 HTTP，不含公网时延和远端服务的缓存。冻结的基线已包含 Fake-IP 与中日韩
修复，因此这不算与某个已发布版本的对比。

完整记录见
[`2026-09-14.json`](https://github.com/Ynewtime/markitai/blob/main/scripts/benchmarks/results/2026-09-14.json)，
复现命令见[基准指南](https://github.com/Ynewtime/markitai/blob/main/scripts/benchmarks/README.md)。

## 排查慢速转换

`-v` 会打印选中的策略和每次回退的原因。`--no-cache` 关闭的是 markitai 自己的
缓存，够不到远端服务的缓存，所以拿已经预热的远端读取器和刚启动的本地 CLI
进程比较，衡量的是不同的工作量。只测量本地转换时用
`--preset minimal --no-remote-fetch`；评估实际使用的完整工作流时，保留平时
开启的模型增强与 OCR 选项。
