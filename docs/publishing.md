# 发布到 PyPI

## 首次设置

### 1. 创建 API Token

1. 登录 PyPI → Account Settings → API tokens
2. 点击 "Add API token"
3. Scope：选择 "Entire account"（首次发布）或指定项目
4. 保存 token（以 `pypi-` 开头）

### 2. 手动首次发布

```bash
# 构建包
uv build --package markitai

# 使用 API token 发布
uv publish --token pypi-xxxxxxxx
```

### 3. 配置 Trusted Publishing

首次发布成功后，配置自动发布：

1. 登录 https://pypi.org
2. 进入项目 **markitai** → **Settings** → **Publishing**
3. 点击 **Add a new publisher**
4. 填写：

| 字段 | 值 |
|-----|-----|
| Owner | `Ynewtime` |
| Repository name | `markitai` |
| Workflow name | `publish.yml` |
| Environment name | （留空） |

5. 点击 **Add**

## 自动发布

### 通过 GitHub Release 触发

1. GitHub → Releases → Create new release
2. 创建新 tag（如 `v0.3.1`）
3. 填写发布说明
4. 点击 **Publish release**

### 手动触发

1. GitHub → Actions → **Publish to PyPI**
2. 点击 **Run workflow**

## 版本管理

发布新版本前：

1. 更新 `packages/markitai/pyproject.toml` 中的版本号
2. 更新 `CHANGELOG.md`
3. 提交更改
4. 创建 Release 或触发工作流

## 故障排除

### 构建失败

```bash
rm -rf dist/
uv build --package markitai
```

### Trusted Publishing 失败

- 确认工作流文件名：`publish.yml`
- 确保仓库 owner/name 正确
- 检查 GitHub Actions 有 `id-token: write` 权限
