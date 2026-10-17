# ChatGPT 订阅

markitai 可以通过你的 ChatGPT 订阅（Codex 登录）发送请求，不必使用 API 密钥。模型名为 `chatgpt/<模型>`，例如 `chatgpt/gpt-6-luna`。不需要安装任何东西：markitai 自己完成登录。

在运行 markitai（或 `markitai serve`）的机器上执行：

```sh
markitai auth chatgpt login
```

它会显示一个网址和一个验证码（OAuth 设备码流程）：在任意浏览器里打开网址，输入验证码并确认。如果还没登录，第一次发出 `chatgpt/` 请求时也会启动登录。`markitai auth chatgpt status` 显示当前账号；之后请在工作台里重新加载模型列表。

如果已经登录过，工作台仍提示 ChatGPT“尚未登录”，说明登录已过期：再运行一次上面的命令。
