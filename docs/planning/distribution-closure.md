# 配送、平台与发布的收尾路径

2026-09-29。本文是可执行收尾计划，不是发布验收结果。R27 是最近完整记录的
CLI/三绑定检查点；R28 的静态 Go 实际结果由协调器单独归档。
当前实现与未完成能力分别见 [CONTROL](../CONTROL.md)、
[bindings](../bindings.md) 和 [remaining-work](../remaining-work.md)。

## 先闭合本机配送

本机已有 Rust、Go、Xcode/SDK、Python 与 Node 工具。以下工作无需新账户或其他
主机，但必须使用冻结源码与新输出目录，保留已有包和失败记录。

1. 使用 [静态构建驱动](../validation/drivers/routing-domains-static-round28/build-static.py)
   保存真实 Cargo 命令、退出码、编译器、完整 metadata、native-static-libs 输出，
   以及 `.a` 和全体受控源文件的字节哈希。前后源码和 clean 状态必须一致。
2. 用 [静态配送脚本](../../scripts/package_go_static.py) 消费该记录。要求归档哈希、
   编译器链接依赖、当前 HEAD 和全部源文件三方闭合；不能只信任一个传入 SHA。
   它生成自足模块/header/archive/许可证/manifest，并真正解包后运行现有 Go
   race 测试，再构建和搬移独立消费者执行。消费者不得依赖仓库 `target`、外置
   Markitai 动态库或动态库搜索环境。
3. 保存最终消费者的 `otool -L` 与 load commands。允许 macOS 系统框架/库；不允许
   Markitai dylib 或开发目录 rpath。静态 Markitai 不等于静态 macOS，也不消除
   Chromium/LibreOffice 等可选运行时。构建产物标记的最低 macOS 版本不等于在该
   版本上执行通过。
4. 对同一最终源码重新构建 CLI、wheel、npm 原生包；从新安装目录执行现有真实
   调用，再验证 manifest 中的实际扩展模块/可执行文件哈希。记录单 CLI、压缩包、
   wheel/npm/Go 模块、最终 Go 消费者分别的体积，不把 `.a` 大小当成用户 executable
   大小，也不混用不同优化 profile。

`markitai_static` 保持可选，默认动态开发方式不变。目前 darwin/arm64 与 linux/amd64（glibc）分支有
配送实现，其他 target 明确拒绝静态模式。C ABI 和 Go 调用接口不需要改变。
Go 开发者构建时仍需 cgo/C linker，最终程序不需要 Rust、Python、Node 或另送
Markitai 动态库。[Rust staticlib 规则](https://doc.rust-lang.org/reference/linkage.html)、
[cgo 包内静态库](https://pkg.go.dev/cmd/cgo) 支持这一区分。

## 许可证闭环

机械收集是输入，不是法律结论。当前脚本保留实际原文及来源路径/哈希，明确输出
`legal_review: not_performed` 和 `unresolved`；其依赖闭包还包括开发、构建和其他
平台条目。不能把“找到一个 LICENSE 文件”或 `unresolved` 为空解释成审查完成。

下一工作包按最终冻结产物逐项处理：

- 为每个缺失文本的依赖，以锁定版本和上游源码身份定位正式许可证/版权文本，
  保存原始下载、URL、revision、哈希和适用关系；不根据 SPDX 名称自己重写原文。
  若上游包无完整文本，保留未解决状态，不能默默套用项目 MIT。
- 区分实际目标的运行时链接闭包、构建工具、开发测试和随包夹具。静态包要包括
  适用 Rust 标准库和 bundled 原生代码文本；wheel/npm/CLI 也要按其实际内容验证。
  当前 workspace metadata 本身不是精确的已链接代码清单。
- 检查 `ring` 等含多来源代码的完整 notice，以及补丁版 PDF reader 的上游许可证
  与修改说明。内嵌 UI 的 Marked/DOMPurify 文本保留在真正包含 UI 的配送物中。
  若配送测试夹具，保留其独立许可证；不能将测试来源误标成原创。
- 为最终每个归档列出可核对的文本路径/哈希及审查结论；存在未解决条目时继续
  保留明确限制。重新解包检查原文完整性，不能只测试 source 目录。

本机可以完成资料归档、完整性检查和差异清单。涉及许可证适用性判断的未解决
事项，需要项目发布负责人明确处理；脚本成功不能替代这一决定。[项目 NOTICE](../../NOTICE)
已有相同边界。

## 必须在外部主机完成的分支

[现有 CI](../../.github/workflows/native.yml) 定义了四个平台和最低 Rust 检查；
定义矩阵不代表这些 job 已执行。按每个平台各自的 clean commit、运行日志、
编译器/SDK、依赖检查及归档哈希记录结果。

| 平台 | 最小独立闭环 | 不能借用的证据 |
|---|---|---|
| Linux x86_64 | 本机构建、CLI及安装 wheel/npm/Go 调用；检查 ELF 动态依赖、glibc 基线、锁/中断/恢复。静态 Go 另加本平台 archive 和编译器实际系统库列表。 | macOS `.a`、`otool` 或一次交叉编译不能证明 Linux 运行。glibc 静态 Markitai 也不是 musl 全静态程序。 |
| Windows x86_64 | 实机 CLI/wheel/npm；路径、进程清理与权限/锁的实际行为。Go/cgo 必须先选定并验证 C 工具链与 Rust target/CRT 的一致组合，再做独立消费。 | 当前 CI 明确跳过 Windows Go 配送；MSVC `.lib` 不能未经验证当作现有 Go C 工具链可直接消费的库。 |
| macOS Intel | 原生 x86_64 构建和安装调用，视觉/OCR/PDF及进程回收分别验证；需要时再制作双架构配送物。 | Rosetta 已保留的失败不能被 arm64 成功覆盖；本机架构切换不等于实体 Intel。 |
| 最低支持系统 | 在实际最低系统安装并运行最终包，包括按需使用的 framework API。 | Mach-O minimum-version 字段和 wheel 文件名只能表达构建声明。 |

先完成基础转换/明确 unsupported 的可运行矩阵，再逐项补平台媒体和非 Unix 恢复。
不要为通过配送测试而隐藏不支持的能力，也不要将平台基础安装通过声称全部功能
一致。当前没有远程运行记录时，应保留 pending；需要配置 Git remote/CI 权限时由
用户或项目负责人指定目标，不能自行选一个远程仓库发布代码。

## 签名与正式发布需要额外身份及授权

本机可以先检查打包内容、记录现有签名状态并准备发布 manifest；这些操作不产生
开发者身份或公证结果。不要为了填写“通过”而生成自签名并称作 Developer ID。

macOS 正式直接分发路径需指定项目所属 Developer ID、采用适当 hardened runtime/
timestamp、提交公证，并保留 Apple 返回的结果和日志；然后在干净消费者环境检验
最终签名后的同一产物。签名改变字节，必须生成新的最终哈希，而不是复用未签名
产物的哈希。`notarytool` 可用不代表已经拥有账户、证书或上传权限。
[Apple 公证要求](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)
与 [Developer ID 身份](https://developer.apple.com/help/account/certificates/create-developer-id-certificates)
是实际前提。静态 `.a` 的完整性记录不等于最终使用者应用的签名。

公开发布前还需确定可访问仓库、正式版本和 Go module import path，以及 npm/PyPI
包名的控制权。准备完成的候选包与验收报告后，再由已有明确授权的发布流程执行
上传；未经指定不能抢占包名、写入远程 release 或使用用户账户发布。最终从实际
发布渠道下载，在新环境安装并核对原始产物身份，才闭合用户安装路径。

PyPI/npm 的 Trusted Publishing 可将发布绑定到指定 CI 身份；配置工作流本身不
是已发布或已有证明。保存并验证实际发布文件的 attestation，不能把它解释成代码
无缺陷或完全可复现构建。[PyPI attestation](https://docs.pypi.org/attestations/producing-attestations/)、
[npm provenance](https://docs.npmjs.com/generating-provenance-statements/) 描述了这一区别。

## 完成顺序与退出标准

顺序为：本机静态消费闭环 → 缺失许可证原文/适用性处理 → 外部原生平台矩阵 →
冻结候选和同质量输入的性能/安装体积对照 → 有授权的签名/发布 → 实际渠道安装。
每项保留失败与缺失，只有对应真实执行通过才更新状态。发布前最后核对
[remaining-work](../remaining-work.md) 的功能主题；配送闭合不等于整个迁移目标完成。

驱动的小型工程收尾也应保留：构建路径清单改用 NUL 分隔，以支持未来带换行或
非 ASCII 的受控文件；超时/启动失败也记录完整 command、退出情况及已产生日志
哈希。当前源码路径没有触发前者，但不能把这个偶然条件变成长期格式假设。
