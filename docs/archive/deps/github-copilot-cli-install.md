---
source:
- https://docs.github.com/api/article/body?pathname=/en/copilot/how-tos/set-up/install-copilot-cli
updated: 2026-01-27
---

# Installing GitHub Copilot CLI

Learn how to install Copilot CLI so that you can use Copilot directly from the command line.

> \[!NOTE]
> GitHub Copilot CLI is in public preview with [data protection](https://gh.io/dpa) and subject to change.

To find out about Copilot CLI before you install it, see [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli).

## Prerequisites

* **An active GitHub Copilot subscription**. See [Copilot plans](https://github.com/features/copilot/plans?ref_product=copilot\&ref_type=engagement\&ref_style=text).
* (On Windows) **PowerShell** v6 or higher

If you have access to GitHub Copilot via your organization or enterprise, you cannot use Copilot CLI if your organization owner or enterprise administrator has disabled it in the organization or enterprise settings. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-github-copilot-features-in-your-organization/managing-policies-for-copilot-in-your-organization).

## Installing or updating Copilot CLI

You can install Copilot CLI using WinGet (Windows), Homebrew (macOS and Linux), npm (all platforms), or an install script (macOS and Linux).

### Installing with WinGet (Windows)

```powershell copy
winget install GitHub.Copilot
```

To install the prerelease version:

```powershell copy
winget install GitHub.Copilot.Prerelease
```

### Installing with Homebrew (macOS and Linux)

```shell copy
brew install copilot-cli
```

To install the prerelease version:

```shell copy
brew install copilot-cli@prerelease
```

### Installing with npm (all platforms, requires Node.js 22+)

```shell copy
npm install -g @github/copilot
```

To install the prerelease version:

```shell copy
npm install -g @github/copilot@prerelease
```

### Installing with the install script (macOS and Linux)

```shell copy
curl -fsSL https://gh.io/copilot-install | bash
```

Or:

```shell copy
wget -qO- https://gh.io/copilot-install | bash
```

To run as root and install to `/usr/local/bin`, use `| sudo bash`.

To install to a custom directory, set the `PREFIX` environment variable. It defaults to `/usr/local` when run as root or `$HOME/.local` when run as a non-root user.

To install a specific version, set the `VERSION` environment variable. It defaults to the latest version.

For example, to install version `v0.0.369` to a custom directory:

```shell copy
curl -fsSL https://gh.io/copilot-install | VERSION="v0.0.369" PREFIX="$HOME/custom" bash
```

### Download from GitHub.com

You can download the executables directly from [the `copilot-cli` repository](https://github.com/github/copilot-cli/releases/).

Download the executable for your platform, unpack it, and run.

## Authenticating with Copilot CLI

On first launch, if you're not currently logged in to GitHub, you'll be prompted to use the `/login` slash command. Enter this command and follow the on-screen instructions to authenticate.

### Authenticating with a personal access token

You can also authenticate using a fine-grained personal access token with the "Copilot Requests" permission enabled.

1. Visit [Fine-grained personal access tokens](https://github.com/settings/personal-access-tokens/new).
2. Under "Permissions," click **Add permissions** and select **Copilot Requests**.
3. Click **Generate token**.
4. Add the token to your environment using the `GH_TOKEN` or `GITHUB_TOKEN` environment variable (in order of precedence).

## Next steps

You can now use Copilot from the command line. See [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli).
