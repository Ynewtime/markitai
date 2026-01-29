---
source:
- https://platform.claude.com/docs/en/agent-sdk/overview.md
- https://platform.claude.com/docs/en/agent-sdk/quickstart.md
- https://platform.claude.com/docs/en/agent-sdk/python.md
---

# Agent SDK overview

Build production AI agents with Claude Code as a library

---

<Note>
The Claude Code SDK has been renamed to the Claude Agent SDK. If you're migrating from the old SDK, see the [Migration Guide](/docs/en/agent-sdk/migration-guide).
</Note>

Build AI agents that autonomously read files, run commands, search the web, edit code, and more. The Agent SDK gives you the same tools, agent loop, and context management that power Claude Code, programmable in Python and TypeScript.

<CodeGroup>
```python Python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    async for message in query(
        prompt="Find and fix the bug in auth.py",
        options=ClaudeAgentOptions(allowed_tools=["Read", "Edit", "Bash"])
    ):
        print(message)  # Claude reads the file, finds the bug, edits it

asyncio.run(main())
```

```typescript TypeScript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Find and fix the bug in auth.py",
  options: { allowedTools: ["Read", "Edit", "Bash"] }
})) {
  console.log(message);  // Claude reads the file, finds the bug, edits it
}
```
</CodeGroup>

The Agent SDK includes built-in tools for reading files, running commands, and editing code, so your agent can start working immediately without you implementing tool execution. Dive into the quickstart or explore real agents built with the SDK:

<CardGroup cols={2}>
  <Card title="Quickstart" icon="play" href="/docs/en/agent-sdk/quickstart">
    Build a bug-fixing agent in minutes
  </Card>
  <Card title="Example agents" icon="star" href="https://github.com/anthropics/claude-agent-sdk-demos">
    Email assistant, research agent, and more
  </Card>
</CardGroup>

## Capabilities

Everything that makes Claude Code powerful is available in the SDK:

<Tabs>
  <Tab title="Built-in tools">
    Your agent can read files, run commands, and search codebases out of the box. Key tools include:

    | Tool | What it does |
    |------|--------------|
    | **Read** | Read any file in the working directory |
    | **Write** | Create new files |
    | **Edit** | Make precise edits to existing files |
    | **Bash** | Run terminal commands, scripts, git operations |
    | **Glob** | Find files by pattern (`**/*.ts`, `src/**/*.py`) |
    | **Grep** | Search file contents with regex |
    | **WebSearch** | Search the web for current information |
    | **WebFetch** | Fetch and parse web page content |
    | **[AskUserQuestion](/docs/en/agent-sdk/user-input#handle-clarifying-questions)** | Ask the user clarifying questions with multiple choice options |

    This example creates an agent that searches your codebase for TODO comments:

    <CodeGroup>
    ```python Python
    import asyncio
    from claude_agent_sdk import query, ClaudeAgentOptions

    async def main():
        async for message in query(
            prompt="Find all TODO comments and create a summary",
            options=ClaudeAgentOptions(allowed_tools=["Read", "Glob", "Grep"])
        ):
            if hasattr(message, "result"):
                print(message.result)

    asyncio.run(main())
    ```

    ```typescript TypeScript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    for await (const message of query({
      prompt: "Find all TODO comments and create a summary",
      options: { allowedTools: ["Read", "Glob", "Grep"] }
    })) {
      if ("result" in message) console.log(message.result);
    }
    ```
    </CodeGroup>

  </Tab>
  <Tab title="Hooks">
    Run custom code at key points in the agent lifecycle. SDK hooks use callback functions to validate, log, block, or transform agent behavior.

    **Available hooks:** `PreToolUse`, `PostToolUse`, `Stop`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, and more.

    This example logs all file changes to an audit file:

    <CodeGroup>
    ```python Python
    import asyncio
    from datetime import datetime
    from claude_agent_sdk import query, ClaudeAgentOptions, HookMatcher

    async def log_file_change(input_data, tool_use_id, context):
        file_path = input_data.get('tool_input', {}).get('file_path', 'unknown')
        with open('./audit.log', 'a') as f:
            f.write(f"{datetime.now()}: modified {file_path}\n")
        return {}

    async def main():
        async for message in query(
            prompt="Refactor utils.py to improve readability",
            options=ClaudeAgentOptions(
                permission_mode="acceptEdits",
                hooks={
                    "PostToolUse": [HookMatcher(matcher="Edit|Write", hooks=[log_file_change])]
                }
            )
        ):
            if hasattr(message, "result"):
                print(message.result)

    asyncio.run(main())
    ```

    ```typescript TypeScript
    import { query, HookCallback } from "@anthropic-ai/claude-agent-sdk";
    import { appendFileSync } from "fs";

    const logFileChange: HookCallback = async (input) => {
      const filePath = (input as any).tool_input?.file_path ?? "unknown";
      appendFileSync("./audit.log", `${new Date().toISOString()}: modified ${filePath}\n`);
      return {};
    };

    for await (const message of query({
      prompt: "Refactor utils.py to improve readability",
      options: {
        permissionMode: "acceptEdits",
        hooks: {
          PostToolUse: [{ matcher: "Edit|Write", hooks: [logFileChange] }]
        }
      }
    })) {
      if ("result" in message) console.log(message.result);
    }
    ```
    </CodeGroup>

    [Learn more about hooks →](/docs/en/agent-sdk/hooks)
  </Tab>
  <Tab title="Subagents">
    Spawn specialized agents to handle focused subtasks. Your main agent delegates work, and subagents report back with results.

    Define custom agents with specialized instructions. Include `Task` in `allowedTools` since subagents are invoked via the Task tool:

    <CodeGroup>
    ```python Python
    import asyncio
    from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

    async def main():
        async for message in query(
            prompt="Use the code-reviewer agent to review this codebase",
            options=ClaudeAgentOptions(
                allowed_tools=["Read", "Glob", "Grep", "Task"],
                agents={
                    "code-reviewer": AgentDefinition(
                        description="Expert code reviewer for quality and security reviews.",
                        prompt="Analyze code quality and suggest improvements.",
                        tools=["Read", "Glob", "Grep"]
                    )
                }
            )
        ):
            if hasattr(message, "result"):
                print(message.result)

    asyncio.run(main())
    ```

    ```typescript TypeScript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    for await (const message of query({
      prompt: "Use the code-reviewer agent to review this codebase",
      options: {
        allowedTools: ["Read", "Glob", "Grep", "Task"],
        agents: {
          "code-reviewer": {
            description: "Expert code reviewer for quality and security reviews.",
            prompt: "Analyze code quality and suggest improvements.",
            tools: ["Read", "Glob", "Grep"]
          }
        }
      }
    })) {
      if ("result" in message) console.log(message.result);
    }
    ```
    </CodeGroup>

    Messages from within a subagent's context include a `parent_tool_use_id` field, letting you track which messages belong to which subagent execution.

    [Learn more about subagents →](/docs/en/agent-sdk/subagents)
  </Tab>
  <Tab title="MCP">
    Connect to external systems via the Model Context Protocol: databases, browsers, APIs, and [hundreds more](https://github.com/modelcontextprotocol/servers).

    This example connects the [Playwright MCP server](https://github.com/microsoft/playwright-mcp) to give your agent browser automation capabilities:

    <CodeGroup>
    ```python Python
    import asyncio
    from claude_agent_sdk import query, ClaudeAgentOptions

    async def main():
        async for message in query(
            prompt="Open example.com and describe what you see",
            options=ClaudeAgentOptions(
                mcp_servers={
                    "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]}
                }
            )
        ):
            if hasattr(message, "result"):
                print(message.result)

    asyncio.run(main())
    ```

    ```typescript TypeScript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    for await (const message of query({
      prompt: "Open example.com and describe what you see",
      options: {
        mcpServers: {
          playwright: { command: "npx", args: ["@playwright/mcp@latest"] }
        }
      }
    })) {
      if ("result" in message) console.log(message.result);
    }
    ```
    </CodeGroup>

    [Learn more about MCP →](/docs/en/agent-sdk/mcp)
  </Tab>
  <Tab title="Permissions">
    Control exactly which tools your agent can use. Allow safe operations, block dangerous ones, or require approval for sensitive actions.

    <Note>
    For interactive approval prompts and the `AskUserQuestion` tool, see [Handle approvals and user input](/docs/en/agent-sdk/user-input).
    </Note>

    This example creates a read-only agent that can analyze but not modify code:

    <CodeGroup>
    ```python Python
    import asyncio
    from claude_agent_sdk import query, ClaudeAgentOptions

    async def main():
        async for message in query(
            prompt="Review this code for best practices",
            options=ClaudeAgentOptions(
                allowed_tools=["Read", "Glob", "Grep"],
                permission_mode="bypassPermissions"
            )
        ):
            if hasattr(message, "result"):
                print(message.result)

    asyncio.run(main())
    ```

    ```typescript TypeScript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    for await (const message of query({
      prompt: "Review this code for best practices",
      options: {
        allowedTools: ["Read", "Glob", "Grep"],
        permissionMode: "bypassPermissions"
      }
    })) {
      if ("result" in message) console.log(message.result);
    }
    ```
    </CodeGroup>

    [Learn more about permissions →](/docs/en/agent-sdk/permissions)
  </Tab>
  <Tab title="Sessions">
    Maintain context across multiple exchanges. Claude remembers files read, analysis done, and conversation history. Resume sessions later, or fork them to explore different approaches.

    This example captures the session ID from the first query, then resumes to continue with full context:

    <CodeGroup>
    ```python Python
    import asyncio
    from claude_agent_sdk import query, ClaudeAgentOptions

    async def main():
        session_id = None

        # First query: capture the session ID
        async for message in query(
            prompt="Read the authentication module",
            options=ClaudeAgentOptions(allowed_tools=["Read", "Glob"])
        ):
            if hasattr(message, 'subtype') and message.subtype == 'init':
                session_id = message.session_id

        # Resume with full context from the first query
        async for message in query(
            prompt="Now find all places that call it",  # "it" = auth module
            options=ClaudeAgentOptions(resume=session_id)
        ):
            if hasattr(message, "result"):
                print(message.result)

    asyncio.run(main())
    ```

    ```typescript TypeScript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    let sessionId: string | undefined;

    // First query: capture the session ID
    for await (const message of query({
      prompt: "Read the authentication module",
      options: { allowedTools: ["Read", "Glob"] }
    })) {
      if (message.type === "system" && message.subtype === "init") {
        sessionId = message.session_id;
      }
    }

    // Resume with full context from the first query
    for await (const message of query({
      prompt: "Now find all places that call it",  // "it" = auth module
      options: { resume: sessionId }
    })) {
      if ("result" in message) console.log(message.result);
    }
    ```
    </CodeGroup>

    [Learn more about sessions →](/docs/en/agent-sdk/sessions)
  </Tab>
</Tabs>

### Claude Code features

The SDK also supports Claude Code's filesystem-based configuration. To use these features, set `setting_sources=["project"]` (Python) or `settingSources: ['project']` (TypeScript)  in your options.

| Feature | Description | Location |
|---------|-------------|----------|
| [Skills](/docs/en/agent-sdk/skills) | Specialized capabilities defined in Markdown | `.claude/skills/SKILL.md` |
| [Slash commands](/docs/en/agent-sdk/slash-commands) | Custom commands for common tasks | `.claude/commands/*.md` |
| [Memory](/docs/en/agent-sdk/modifying-system-prompts) | Project context and instructions | `CLAUDE.md` or `.claude/CLAUDE.md` |
| [Plugins](/docs/en/agent-sdk/plugins) | Extend with custom commands, agents, and MCP servers | Programmatic via `plugins` option |

## Get started

<Steps>
  <Step title="Install Claude Code">
    The SDK uses Claude Code as its runtime:

    <Tabs>
      <Tab title="macOS/Linux/WSL">
        ```bash
        curl -fsSL https://claude.ai/install.sh | bash
        ```
      </Tab>
      <Tab title="Homebrew">
        ```bash
        brew install --cask claude-code
        ```
      </Tab>
      <Tab title="WinGet">
        ```powershell
        winget install Anthropic.ClaudeCode
        ```
      </Tab>
    </Tabs>

    See [Claude Code setup](https://code.claude.com/docs/en/setup) for Windows and other options.
  </Step>
  <Step title="Install the SDK">
    <Tabs>
      <Tab title="TypeScript">
        ```bash
        npm install @anthropic-ai/claude-agent-sdk
        ```
      </Tab>
      <Tab title="Python">
        ```bash
        pip install claude-agent-sdk
        ```
      </Tab>
    </Tabs>
  </Step>
  <Step title="Set your API key">
    ```bash
    export ANTHROPIC_API_KEY=your-api-key
    ```
    Get your key from the [Console](https://platform.claude.com/).

    The SDK also supports authentication via third-party API providers:

    - **Amazon Bedrock**: set `CLAUDE_CODE_USE_BEDROCK=1` environment variable and configure AWS credentials
    - **Google Vertex AI**: set `CLAUDE_CODE_USE_VERTEX=1` environment variable and configure Google Cloud credentials
    - **Microsoft Foundry**: set `CLAUDE_CODE_USE_FOUNDRY=1` environment variable and configure Azure credentials

    <Note>
    Unless previously approved, we do not allow third party developers to offer Claude.ai login or rate limits for their products, including agents built on the Claude Agent SDK. Please use the API key authentication methods described in this document instead.
    </Note>
  </Step>
  <Step title="Run your first agent">
    This example creates an agent that lists files in your current directory using built-in tools.

    <CodeGroup>
    ```python Python
    import asyncio
    from claude_agent_sdk import query, ClaudeAgentOptions

    async def main():
        async for message in query(
            prompt="What files are in this directory?",
            options=ClaudeAgentOptions(allowed_tools=["Bash", "Glob"])
        ):
            if hasattr(message, "result"):
                print(message.result)

    asyncio.run(main())
    ```

    ```typescript TypeScript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    for await (const message of query({
      prompt: "What files are in this directory?",
      options: { allowedTools: ["Bash", "Glob"] },
    })) {
      if ("result" in message) console.log(message.result);
    }
    ```
    </CodeGroup>
  </Step>
</Steps>

**Ready to build?** Follow the [Quickstart](/docs/en/agent-sdk/quickstart) to create an agent that finds and fixes bugs in minutes.

## Compare the Agent SDK to other Claude tools

The Claude platform offers multiple ways to build with Claude. Here's how the Agent SDK fits in:

<Tabs>
  <Tab title="Agent SDK vs Client SDK">
    The [Anthropic Client SDK](/docs/en/api/client-sdks) gives you direct API access: you send prompts and implement tool execution yourself. The **Agent SDK** gives you Claude with built-in tool execution.

    With the Client SDK, you implement a tool loop. With the Agent SDK, Claude handles it:

    <CodeGroup>
    ```python Python
    # Client SDK: You implement the tool loop
    response = client.messages.create(...)
    while response.stop_reason == "tool_use":
        result = your_tool_executor(response.tool_use)
        response = client.messages.create(tool_result=result, ...)

    # Agent SDK: Claude handles tools autonomously
    async for message in query(prompt="Fix the bug in auth.py"):
        print(message)
    ```

    ```typescript TypeScript
    // Client SDK: You implement the tool loop
    let response = await client.messages.create({...});
    while (response.stop_reason === "tool_use") {
      const result = yourToolExecutor(response.tool_use);
      response = await client.messages.create({ tool_result: result, ... });
    }

    // Agent SDK: Claude handles tools autonomously
    for await (const message of query({ prompt: "Fix the bug in auth.py" })) {
      console.log(message);
    }
    ```
    </CodeGroup>
  </Tab>
  <Tab title="Agent SDK vs Claude Code CLI">
    Same capabilities, different interface:

    | Use case | Best choice |
    |----------|-------------|
    | Interactive development | CLI |
    | CI/CD pipelines | SDK |
    | Custom applications | SDK |
    | One-off tasks | CLI |
    | Production automation | SDK |

    Many teams use both: CLI for daily development, SDK for production. Workflows translate directly between them.
  </Tab>
</Tabs>

## Changelog

View the full changelog for SDK updates, bug fixes, and new features:

- **TypeScript SDK**: [view CHANGELOG.md](https://github.com/anthropics/claude-agent-sdk-typescript/blob/main/CHANGELOG.md)
- **Python SDK**: [view CHANGELOG.md](https://github.com/anthropics/claude-agent-sdk-python/blob/main/CHANGELOG.md)

## Reporting bugs

If you encounter bugs or issues with the Agent SDK:

- **TypeScript SDK**: [report issues on GitHub](https://github.com/anthropics/claude-agent-sdk-typescript/issues)
- **Python SDK**: [report issues on GitHub](https://github.com/anthropics/claude-agent-sdk-python/issues)

## Branding guidelines

For partners integrating the Claude Agent SDK, use of Claude branding is optional. When referencing Claude in your product:

**Allowed:**
- "Claude Agent" (preferred for dropdown menus)
- "Claude" (when within a menu already labeled "Agents")
- "{YourAgentName} Powered by Claude" (if you have an existing agent name)

**Not permitted:**
- "Claude Code" or "Claude Code Agent"
- Claude Code-branded ASCII art or visual elements that mimic Claude Code

Your product should maintain its own branding and not appear to be Claude Code or any Anthropic product. For questions about branding compliance, contact our [sales team](https://www.anthropic.com/contact-sales).

## License and terms

Use of the Claude Agent SDK is governed by [Anthropic's Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms), including when you use it to power products and services that you make available to your own customers and end users, except to the extent a specific component or dependency is covered by a different license as indicated in that component's LICENSE file.

## Next steps

<CardGroup cols={2}>
  <Card title="Quickstart" icon="play" href="/docs/en/agent-sdk/quickstart">
    Build an agent that finds and fixes bugs in minutes
  </Card>
  <Card title="Example agents" icon="star" href="https://github.com/anthropics/claude-agent-sdk-demos">
    Email assistant, research agent, and more
  </Card>
  <Card title="TypeScript SDK" icon="code" href="/docs/en/agent-sdk/typescript">
    Full TypeScript API reference and examples
  </Card>
  <Card title="Python SDK" icon="code" href="/docs/en/agent-sdk/python">
    Full Python API reference and examples
  </Card>
</CardGroup>


---


# Quickstart

Get started with the Python or TypeScript Agent SDK to build AI agents that work autonomously

---

Use the Agent SDK to build an AI agent that reads your code, finds bugs, and fixes them, all without manual intervention.

**What you'll do:**
1. Set up a project with the Agent SDK
2. Create a file with some buggy code
3. Run an agent that finds and fixes the bugs automatically

## Prerequisites

- **Node.js 18+** or **Python 3.10+**
- An **Anthropic account** ([sign up here](https://platform.claude.com/))

## Setup

<Steps>
  <Step title="Install Claude Code">
    The Agent SDK uses Claude Code as its runtime. Install it for your platform:

    <Tabs>
      <Tab title="macOS/Linux/WSL">
        ```bash
        curl -fsSL https://claude.ai/install.sh | bash
        ```
      </Tab>
      <Tab title="Homebrew">
        ```bash
        brew install --cask claude-code
        ```
      </Tab>
      <Tab title="WinGet">
        ```powershell
        winget install Anthropic.ClaudeCode
        ```
      </Tab>
    </Tabs>

    After installing Claude Code onto your machine, run `claude` in your terminal and follow the prompts to authenticate. The SDK will use this authentication automatically.

    <Tip>
    For more information on Claude Code installation, see [Claude Code setup](https://code.claude.com/docs/en/setup).
    </Tip>
  </Step>

  <Step title="Create a project folder">
    Create a new directory for this quickstart:

    ```bash
    mkdir my-agent && cd my-agent
    ```

    For your own projects, you can run the SDK from any folder; it will have access to files in that directory and its subdirectories by default.
  </Step>

  <Step title="Install the SDK">
    Install the Agent SDK package for your language:

    <Tabs>
      <Tab title="TypeScript">
        ```bash
        npm install @anthropic-ai/claude-agent-sdk
        ```
      </Tab>
      <Tab title="Python (uv)">
        [uv Python package manager](https://docs.astral.sh/uv/) is a fast Python package manager that handles virtual environments automatically:
        ```bash
        uv init && uv add claude-agent-sdk
        ```
      </Tab>
      <Tab title="Python (pip)">
        Create a virtual environment first, then install:
        ```bash
        python3 -m venv .venv && source .venv/bin/activate
        pip3 install claude-agent-sdk
        ```
      </Tab>
    </Tabs>
  </Step>

  <Step title="Set your API key">
    If you've already authenticated Claude Code (by running `claude` in your terminal), the SDK uses that authentication automatically.

    Otherwise, you need an API key, which you can get from the [Claude Console](https://platform.claude.com/).

    Create a `.env` file in your project directory and store the API key there:

    ```bash
    ANTHROPIC_API_KEY=your-api-key
    ```

    <Note>
    **Using Amazon Bedrock, Google Vertex AI, or Microsoft Azure?** See the setup guides for [Bedrock](https://code.claude.com/docs/en/amazon-bedrock), [Vertex AI](https://code.claude.com/docs/en/google-vertex-ai), or [Azure AI Foundry](https://code.claude.com/docs/en/azure-ai-foundry).

    Unless previously approved, Anthropic does not allow third party developers to offer claude.ai login or rate limits for their products, including agents built on the Claude Agent SDK. Please use the API key authentication methods described in this document instead.
    </Note>
  </Step>
</Steps>

## Create a buggy file

This quickstart walks you through building an agent that can find and fix bugs in code. First, you need a file with some intentional bugs for the agent to fix. Create `utils.py` in the `my-agent` directory and paste the following code:

```python
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

def get_user_name(user):
    return user["name"].upper()
```

This code has two bugs:
1. `calculate_average([])` crashes with division by zero
2. `get_user_name(None)` crashes with a TypeError

## Build an agent that finds and fixes bugs

Create `agent.py` if you're using the Python SDK, or `agent.ts` for TypeScript:

<CodeGroup>
```python Python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

async def main():
    # Agentic loop: streams messages as Claude works
    async for message in query(
        prompt="Review utils.py for bugs that would cause crashes. Fix any issues you find.",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Glob"],  # Tools Claude can use
            permission_mode="acceptEdits"            # Auto-approve file edits
        )
    ):
        # Print human-readable output
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)              # Claude's reasoning
                elif hasattr(block, "name"):
                    print(f"Tool: {block.name}")   # Tool being called
        elif isinstance(message, ResultMessage):
            print(f"Done: {message.subtype}")      # Final result

asyncio.run(main())
```

```typescript TypeScript
import { query } from "@anthropic-ai/claude-agent-sdk";

// Agentic loop: streams messages as Claude works
for await (const message of query({
  prompt: "Review utils.py for bugs that would cause crashes. Fix any issues you find.",
  options: {
    allowedTools: ["Read", "Edit", "Glob"],  // Tools Claude can use
    permissionMode: "acceptEdits"            // Auto-approve file edits
  }
})) {
  // Print human-readable output
  if (message.type === "assistant" && message.message?.content) {
    for (const block of message.message.content) {
      if ("text" in block) {
        console.log(block.text);             // Claude's reasoning
      } else if ("name" in block) {
        console.log(`Tool: ${block.name}`);  // Tool being called
      }
    }
  } else if (message.type === "result") {
    console.log(`Done: ${message.subtype}`); // Final result
  }
}
```
</CodeGroup>

This code has three main parts:

1. **`query`**: the main entry point that creates the agentic loop. It returns an async iterator, so you use `async for` to stream messages as Claude works. See the full API in the [Python](/docs/en/agent-sdk/python#query) or [TypeScript](/docs/en/agent-sdk/typescript#query) SDK reference.

2. **`prompt`**: what you want Claude to do. Claude figures out which tools to use based on the task.

3. **`options`**: configuration for the agent. This example uses `allowedTools` to restrict Claude to `Read`, `Edit`, and `Glob`, and `permissionMode: "acceptEdits"` to auto-approve file changes. Other options include `systemPrompt`, `mcpServers`, and more. See all options for [Python](/docs/en/agent-sdk/python#claudeagentoptions) or [TypeScript](/docs/en/agent-sdk/typescript#claudeagentoptions).

The `async for` loop keeps running as Claude thinks, calls tools, observes results, and decides what to do next. Each iteration yields a message: Claude's reasoning, a tool call, a tool result, or the final outcome. The SDK handles the orchestration (tool execution, context management, retries) so you just consume the stream. The loop ends when Claude finishes the task or hits an error.

The message handling inside the loop filters for human-readable output. Without filtering, you'd see raw message objects including system initialization and internal state, which is useful for debugging but noisy otherwise.

<Note>
This example uses streaming to show progress in real-time. If you don't need live output (e.g., for background jobs or CI pipelines), you can collect all messages at once. See [Streaming vs. single-turn mode](/docs/en/agent-sdk/streaming-vs-single-mode) for details.
</Note>

### Run your agent

Your agent is ready. Run it with the following command:

<Tabs>
  <Tab title="Python">
    ```bash
    python3 agent.py
    ```
  </Tab>
  <Tab title="TypeScript">
    ```bash
    npx tsx agent.ts
    ```
  </Tab>
</Tabs>

After running, check `utils.py`. You'll see defensive code handling empty lists and null users. Your agent autonomously:

1. **Read** `utils.py` to understand the code
2. **Analyzed** the logic and identified edge cases that would crash
3. **Edited** the file to add proper error handling

This is what makes the Agent SDK different: Claude executes tools directly instead of asking you to implement them.

<Note>
If you see "Claude Code not found", [install Claude Code](#install-claude-code) and restart your terminal. For "API key not found", [set your API key](#set-your-api-key). See the [full troubleshooting guide](https://code.claude.com/docs/en/troubleshooting) for more help.
</Note>

### Try other prompts

Now that your agent is set up, try some different prompts:

- `"Add docstrings to all functions in utils.py"`
- `"Add type hints to all functions in utils.py"`
- `"Create a README.md documenting the functions in utils.py"`

### Customize your agent

You can modify your agent's behavior by changing the options. Here are a few examples:

**Add web search capability:**

<CodeGroup>
```python Python
options=ClaudeAgentOptions(
    allowed_tools=["Read", "Edit", "Glob", "WebSearch"],
    permission_mode="acceptEdits"
)
```

```typescript TypeScript
options: {
  allowedTools: ["Read", "Edit", "Glob", "WebSearch"],
  permissionMode: "acceptEdits"
}
```
</CodeGroup>

**Give Claude a custom system prompt:**

<CodeGroup>
```python Python
options=ClaudeAgentOptions(
    allowed_tools=["Read", "Edit", "Glob"],
    permission_mode="acceptEdits",
    system_prompt="You are a senior Python developer. Always follow PEP 8 style guidelines."
)
```

```typescript TypeScript
options: {
  allowedTools: ["Read", "Edit", "Glob"],
  permissionMode: "acceptEdits",
  systemPrompt: "You are a senior Python developer. Always follow PEP 8 style guidelines."
}
```
</CodeGroup>

**Run commands in the terminal:**

<CodeGroup>
```python Python
options=ClaudeAgentOptions(
    allowed_tools=["Read", "Edit", "Glob", "Bash"],
    permission_mode="acceptEdits"
)
```

```typescript TypeScript
options: {
  allowedTools: ["Read", "Edit", "Glob", "Bash"],
  permissionMode: "acceptEdits"
}
```
</CodeGroup>

With `Bash` enabled, try: `"Write unit tests for utils.py, run them, and fix any failures"`

## Key concepts

**Tools** control what your agent can do:

| Tools | What the agent can do |
|-------|----------------------|
| `Read`, `Glob`, `Grep` | Read-only analysis |
| `Read`, `Edit`, `Glob` | Analyze and modify code |
| `Read`, `Edit`, `Bash`, `Glob`, `Grep` | Full automation |

**Permission modes** control how much human oversight you want:

| Mode | Behavior | Use case |
|------|----------|----------|
| `acceptEdits` | Auto-approves file edits, asks for other actions | Trusted development workflows |
| `bypassPermissions` | Runs without prompts | CI/CD pipelines, automation |
| `default` | Requires a `canUseTool` callback to handle approval | Custom approval flows |

The example above uses `acceptEdits` mode, which auto-approves file operations so the agent can run without interactive prompts. If you want to prompt users for approval, use `default` mode and provide a [`canUseTool` callback](/docs/en/agent-sdk/user-input) that collects user input. For more control, see [Permissions](/docs/en/agent-sdk/permissions).

## Next steps

Now that you've created your first agent, learn how to extend its capabilities and tailor it to your use case:

- **[Permissions](/docs/en/agent-sdk/permissions)**: control what your agent can do and when it needs approval
- **[Hooks](/docs/en/agent-sdk/hooks)**: run custom code before or after tool calls
- **[Sessions](/docs/en/agent-sdk/sessions)**: build multi-turn agents that maintain context
- **[MCP servers](/docs/en/agent-sdk/mcp)**: connect to databases, browsers, APIs, and other external systems
- **[Hosting](/docs/en/agent-sdk/hosting)**: deploy agents to Docker, cloud, and CI/CD
- **[Example agents](https://github.com/anthropics/claude-agent-sdk-demos)**: see complete examples: email assistant, research agent, and more


---


# Agent SDK reference - Python

Complete API reference for the Python Agent SDK, including all functions, types, and classes.

---

## Installation

```bash
pip install claude-agent-sdk
```

## Choosing Between `query()` and `ClaudeSDKClient`

The Python SDK provides two ways to interact with Claude Code:

### Quick Comparison

| Feature             | `query()`                     | `ClaudeSDKClient`                  |
| :------------------ | :---------------------------- | :--------------------------------- |
| **Session**         | Creates new session each time | Reuses same session                |
| **Conversation**    | Single exchange               | Multiple exchanges in same context |
| **Connection**      | Managed automatically         | Manual control                     |
| **Streaming Input** | ✅ Supported                  | ✅ Supported                       |
| **Interrupts**      | ❌ Not supported              | ✅ Supported                       |
| **Hooks**           | ❌ Not supported              | ✅ Supported                       |
| **Custom Tools**    | ❌ Not supported              | ✅ Supported                       |
| **Continue Chat**   | ❌ New session each time      | ✅ Maintains conversation          |
| **Use Case**        | One-off tasks                 | Continuous conversations           |

### When to Use `query()` (New Session Each Time)

**Best for:**

- One-off questions where you don't need conversation history
- Independent tasks that don't require context from previous exchanges
- Simple automation scripts
- When you want a fresh start each time

### When to Use `ClaudeSDKClient` (Continuous Conversation)

**Best for:**

- **Continuing conversations** - When you need Claude to remember context
- **Follow-up questions** - Building on previous responses
- **Interactive applications** - Chat interfaces, REPLs
- **Response-driven logic** - When next action depends on Claude's response
- **Session control** - Managing conversation lifecycle explicitly

## Functions

### `query()`

Creates a new session for each interaction with Claude Code. Returns an async iterator that yields messages as they arrive. Each call to `query()` starts fresh with no memory of previous interactions.

```python
async def query(
    *,
    prompt: str | AsyncIterable[dict[str, Any]],
    options: ClaudeAgentOptions | None = None
) -> AsyncIterator[Message]
```

#### Parameters

| Parameter | Type                         | Description                                                                |
| :-------- | :--------------------------- | :------------------------------------------------------------------------- |
| `prompt`  | `str \| AsyncIterable[dict]` | The input prompt as a string or async iterable for streaming mode          |
| `options` | `ClaudeAgentOptions \| None` | Optional configuration object (defaults to `ClaudeAgentOptions()` if None) |

#### Returns

Returns an `AsyncIterator[Message]` that yields messages from the conversation.

#### Example - With options

```python

import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(
        system_prompt="You are an expert Python developer",
        permission_mode='acceptEdits',
        cwd="/home/user/project"
    )

    async for message in query(
        prompt="Create a Python web server",
        options=options
    ):
        print(message)


asyncio.run(main())
```

### `tool()`

Decorator for defining MCP tools with type safety.

```python
def tool(
    name: str,
    description: str,
    input_schema: type | dict[str, Any]
) -> Callable[[Callable[[Any], Awaitable[dict[str, Any]]]], SdkMcpTool[Any]]
```

#### Parameters

| Parameter      | Type                     | Description                                             |
| :------------- | :----------------------- | :------------------------------------------------------ |
| `name`         | `str`                    | Unique identifier for the tool                          |
| `description`  | `str`                    | Human-readable description of what the tool does        |
| `input_schema` | `type \| dict[str, Any]` | Schema defining the tool's input parameters (see below) |

#### Input Schema Options

1. **Simple type mapping** (recommended):

   ```python
   {"text": str, "count": int, "enabled": bool}
   ```

2. **JSON Schema format** (for complex validation):
   ```python
   {
       "type": "object",
       "properties": {
           "text": {"type": "string"},
           "count": {"type": "integer", "minimum": 0}
       },
       "required": ["text"]
   }
   ```

#### Returns

A decorator function that wraps the tool implementation and returns an `SdkMcpTool` instance.

#### Example

```python
from claude_agent_sdk import tool
from typing import Any

@tool("greet", "Greet a user", {"name": str})
async def greet(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{
            "type": "text",
            "text": f"Hello, {args['name']}!"
        }]
    }
```

### `create_sdk_mcp_server()`

Create an in-process MCP server that runs within your Python application.

```python
def create_sdk_mcp_server(
    name: str,
    version: str = "1.0.0",
    tools: list[SdkMcpTool[Any]] | None = None
) -> McpSdkServerConfig
```

#### Parameters

| Parameter | Type                            | Default   | Description                                           |
| :-------- | :------------------------------ | :-------- | :---------------------------------------------------- |
| `name`    | `str`                           | -         | Unique identifier for the server                      |
| `version` | `str`                           | `"1.0.0"` | Server version string                                 |
| `tools`   | `list[SdkMcpTool[Any]] \| None` | `None`    | List of tool functions created with `@tool` decorator |

#### Returns

Returns an `McpSdkServerConfig` object that can be passed to `ClaudeAgentOptions.mcp_servers`.

#### Example

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args):
    return {
        "content": [{
            "type": "text",
            "text": f"Sum: {args['a'] + args['b']}"
        }]
    }

@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply(args):
    return {
        "content": [{
            "type": "text",
            "text": f"Product: {args['a'] * args['b']}"
        }]
    }

calculator = create_sdk_mcp_server(
    name="calculator",
    version="2.0.0",
    tools=[add, multiply]  # Pass decorated functions
)

# Use with Claude
options = ClaudeAgentOptions(
    mcp_servers={"calc": calculator},
    allowed_tools=["mcp__calc__add", "mcp__calc__multiply"]
)
```

## Classes

### `ClaudeSDKClient`

**Maintains a conversation session across multiple exchanges.** This is the Python equivalent of how the TypeScript SDK's `query()` function works internally - it creates a client object that can continue conversations.

#### Key Features

- **Session Continuity**: Maintains conversation context across multiple `query()` calls
- **Same Conversation**: Claude remembers previous messages in the session
- **Interrupt Support**: Can stop Claude mid-execution
- **Explicit Lifecycle**: You control when the session starts and ends
- **Response-driven Flow**: Can react to responses and send follow-ups
- **Custom Tools & Hooks**: Supports custom tools (created with `@tool` decorator) and hooks

```python
class ClaudeSDKClient:
    def __init__(self, options: ClaudeAgentOptions | None = None)
    async def connect(self, prompt: str | AsyncIterable[dict] | None = None) -> None
    async def query(self, prompt: str | AsyncIterable[dict], session_id: str = "default") -> None
    async def receive_messages(self) -> AsyncIterator[Message]
    async def receive_response(self) -> AsyncIterator[Message]
    async def interrupt(self) -> None
    async def rewind_files(self, user_message_uuid: str) -> None
    async def disconnect(self) -> None
```

#### Methods

| Method                      | Description                                                         |
| :-------------------------- | :------------------------------------------------------------------ |
| `__init__(options)`         | Initialize the client with optional configuration                   |
| `connect(prompt)`           | Connect to Claude with an optional initial prompt or message stream |
| `query(prompt, session_id)` | Send a new request in streaming mode                                |
| `receive_messages()`        | Receive all messages from Claude as an async iterator               |
| `receive_response()`        | Receive messages until and including a ResultMessage                |
| `interrupt()`               | Send interrupt signal (only works in streaming mode)                |
| `rewind_files(user_message_uuid)` | Restore files to their state at the specified user message. Requires `enable_file_checkpointing=True`. See [File checkpointing](/docs/en/agent-sdk/file-checkpointing) |
| `disconnect()`              | Disconnect from Claude                                              |

#### Context Manager Support

The client can be used as an async context manager for automatic connection management:

```python
async with ClaudeSDKClient() as client:
    await client.query("Hello Claude")
    async for message in client.receive_response():
        print(message)
```

> **Important:** When iterating over messages, avoid using `break` to exit early as this can cause asyncio cleanup issues. Instead, let the iteration complete naturally or use flags to track when you've found what you need.

#### Example - Continuing a conversation

```python
import asyncio
from claude_agent_sdk import ClaudeSDKClient, AssistantMessage, TextBlock, ResultMessage

async def main():
    async with ClaudeSDKClient() as client:
        # First question
        await client.query("What's the capital of France?")

        # Process response
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")

        # Follow-up question - Claude remembers the previous context
        await client.query("What's the population of that city?")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")

        # Another follow-up - still in the same conversation
        await client.query("What are some famous landmarks there?")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")

asyncio.run(main())
```

#### Example - Streaming input with ClaudeSDKClient

```python
import asyncio
from claude_agent_sdk import ClaudeSDKClient

async def message_stream():
    """Generate messages dynamically."""
    yield {"type": "text", "text": "Analyze the following data:"}
    await asyncio.sleep(0.5)
    yield {"type": "text", "text": "Temperature: 25°C"}
    await asyncio.sleep(0.5)
    yield {"type": "text", "text": "Humidity: 60%"}
    await asyncio.sleep(0.5)
    yield {"type": "text", "text": "What patterns do you see?"}

async def main():
    async with ClaudeSDKClient() as client:
        # Stream input to Claude
        await client.query(message_stream())

        # Process response
        async for message in client.receive_response():
            print(message)

        # Follow-up in same session
        await client.query("Should we be concerned about these readings?")

        async for message in client.receive_response():
            print(message)

asyncio.run(main())
```

#### Example - Using interrupts

```python
import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def interruptible_task():
    options = ClaudeAgentOptions(
        allowed_tools=["Bash"],
        permission_mode="acceptEdits"
    )

    async with ClaudeSDKClient(options=options) as client:
        # Start a long-running task
        await client.query("Count from 1 to 100 slowly")

        # Let it run for a bit
        await asyncio.sleep(2)

        # Interrupt the task
        await client.interrupt()
        print("Task interrupted!")

        # Send a new command
        await client.query("Just say hello instead")

        async for message in client.receive_response():
            # Process the new response
            pass

asyncio.run(interruptible_task())
```

#### Example - Advanced permission control

```python
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions
)
from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

async def custom_permission_handler(
    tool_name: str,
    input_data: dict,
    context: dict
) -> PermissionResultAllow | PermissionResultDeny:
    """Custom logic for tool permissions."""

    # Block writes to system directories
    if tool_name == "Write" and input_data.get("file_path", "").startswith("/system/"):
        return PermissionResultDeny(
            message="System directory write not allowed",
            interrupt=True
        )

    # Redirect sensitive file operations
    if tool_name in ["Write", "Edit"] and "config" in input_data.get("file_path", ""):
        safe_path = f"./sandbox/{input_data['file_path']}"
        return PermissionResultAllow(
            updated_input={**input_data, "file_path": safe_path}
        )

    # Allow everything else
    return PermissionResultAllow(updated_input=input_data)

async def main():
    options = ClaudeAgentOptions(
        can_use_tool=custom_permission_handler,
        allowed_tools=["Read", "Write", "Edit"]
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Update the system config file")

        async for message in client.receive_response():
            # Will use sandbox path instead
            print(message)

asyncio.run(main())
```

## Types

### `SdkMcpTool`

Definition for an SDK MCP tool created with the `@tool` decorator.

```python
@dataclass
class SdkMcpTool(Generic[T]):
    name: str
    description: str
    input_schema: type[T] | dict[str, Any]
    handler: Callable[[T], Awaitable[dict[str, Any]]]
```

| Property       | Type                                       | Description                                |
| :------------- | :----------------------------------------- | :----------------------------------------- |
| `name`         | `str`                                      | Unique identifier for the tool             |
| `description`  | `str`                                      | Human-readable description                 |
| `input_schema` | `type[T] \| dict[str, Any]`                | Schema for input validation                |
| `handler`      | `Callable[[T], Awaitable[dict[str, Any]]]` | Async function that handles tool execution |

### `ClaudeAgentOptions`

Configuration dataclass for Claude Code queries.

```python
@dataclass
class ClaudeAgentOptions:
    tools: list[str] | ToolsPreset | None = None
    allowed_tools: list[str] = field(default_factory=list)
    system_prompt: str | SystemPromptPreset | None = None
    mcp_servers: dict[str, McpServerConfig] | str | Path = field(default_factory=dict)
    permission_mode: PermissionMode | None = None
    continue_conversation: bool = False
    resume: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    fallback_model: str | None = None
    betas: list[SdkBeta] = field(default_factory=list)
    output_format: OutputFormat | None = None
    permission_prompt_tool_name: str | None = None
    cwd: str | Path | None = None
    cli_path: str | Path | None = None
    settings: str | None = None
    add_dirs: list[str | Path] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, str | None] = field(default_factory=dict)
    max_buffer_size: int | None = None
    debug_stderr: Any = sys.stderr  # Deprecated
    stderr: Callable[[str], None] | None = None
    can_use_tool: CanUseTool | None = None
    hooks: dict[HookEvent, list[HookMatcher]] | None = None
    user: str | None = None
    include_partial_messages: bool = False
    fork_session: bool = False
    agents: dict[str, AgentDefinition] | None = None
    setting_sources: list[SettingSource] | None = None
    max_thinking_tokens: int | None = None
```

| Property                      | Type                                         | Default              | Description                                                                                                                                                                             |
| :---------------------------- | :------------------------------------------- | :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tools`                       | `list[str] \| ToolsPreset \| None`           | `None`               | Tools configuration. Use `{"type": "preset", "preset": "claude_code"}` for Claude Code's default tools                                                                                  |
| `allowed_tools`               | `list[str]`                                  | `[]`                 | List of allowed tool names                                                                                                                                                              |
| `system_prompt`               | `str \| SystemPromptPreset \| None`          | `None`               | System prompt configuration. Pass a string for custom prompt, or use `{"type": "preset", "preset": "claude_code"}` for Claude Code's system prompt. Add `"append"` to extend the preset |
| `mcp_servers`                 | `dict[str, McpServerConfig] \| str \| Path`  | `{}`                 | MCP server configurations or path to config file                                                                                                                                        |
| `permission_mode`             | `PermissionMode \| None`                     | `None`               | Permission mode for tool usage                                                                                                                                                          |
| `continue_conversation`       | `bool`                                       | `False`              | Continue the most recent conversation                                                                                                                                                   |
| `resume`                      | `str \| None`                                | `None`               | Session ID to resume                                                                                                                                                                    |
| `max_turns`                   | `int \| None`                                | `None`               | Maximum conversation turns                                                                                                                                                              |
| `max_budget_usd`              | `float \| None`                              | `None`               | Maximum budget in USD for the session                                                                                                                                                   |
| `disallowed_tools`            | `list[str]`                                  | `[]`                 | List of disallowed tool names                                                                                                                                                           |
| `enable_file_checkpointing`   | `bool`                                       | `False`              | Enable file change tracking for rewinding. See [File checkpointing](/docs/en/agent-sdk/file-checkpointing)                                                                              |
| `model`                       | `str \| None`                                | `None`               | Claude model to use                                                                                                                                                                     |
| `fallback_model`              | `str \| None`                                | `None`               | Fallback model to use if the primary model fails                                                                                                                                        |
| `betas`                       | `list[SdkBeta]`                              | `[]`                 | Beta features to enable. See [`SdkBeta`](#sdkbeta) for available options                                                                                                                |
| `output_format`               | [`OutputFormat`](#outputformat) ` \| None`   | `None`               | Define output format for agent results. See [Structured outputs](/docs/en/agent-sdk/structured-outputs) for details                                                                    |
| `permission_prompt_tool_name` | `str \| None`                                | `None`               | MCP tool name for permission prompts                                                                                                                                                    |
| `cwd`                         | `str \| Path \| None`                        | `None`               | Current working directory                                                                                                                                                               |
| `cli_path`                    | `str \| Path \| None`                        | `None`               | Custom path to the Claude Code CLI executable                                                                                                                                           |
| `settings`                    | `str \| None`                                | `None`               | Path to settings file                                                                                                                                                                   |
| `add_dirs`                    | `list[str \| Path]`                          | `[]`                 | Additional directories Claude can access                                                                                                                                                |
| `env`                         | `dict[str, str]`                             | `{}`                 | Environment variables                                                                                                                                                                   |
| `extra_args`                  | `dict[str, str \| None]`                     | `{}`                 | Additional CLI arguments to pass directly to the CLI                                                                                                                                    |
| `max_buffer_size`             | `int \| None`                                | `None`               | Maximum bytes when buffering CLI stdout                                                                                                                                                 |
| `debug_stderr`                | `Any`                                        | `sys.stderr`         | _Deprecated_ - File-like object for debug output. Use `stderr` callback instead                                                                                                         |
| `stderr`                      | `Callable[[str], None] \| None`              | `None`               | Callback function for stderr output from CLI                                                                                                                                            |
| `can_use_tool`                | [`CanUseTool`](#canusertool) ` \| None`      | `None`               | Tool permission callback function. See [Permission types](#canusertool) for details                                                                                                     |
| `hooks`                       | `dict[HookEvent, list[HookMatcher]] \| None` | `None`               | Hook configurations for intercepting events                                                                                                                                             |
| `user`                        | `str \| None`                                | `None`               | User identifier                                                                                                                                                                         |
| `include_partial_messages`    | `bool`                                       | `False`              | Include partial message streaming events. When enabled, [`StreamEvent`](#streamevent) messages are yielded                                                                              |
| `fork_session`                | `bool`                                       | `False`              | When resuming with `resume`, fork to a new session ID instead of continuing the original session                                                                                        |
| `agents`                      | `dict[str, AgentDefinition] \| None`         | `None`               | Programmatically defined subagents                                                                                                                                                      |
| `plugins`                     | `list[SdkPluginConfig]`                      | `[]`                 | Load custom plugins from local paths. See [Plugins](/docs/en/agent-sdk/plugins) for details                                                                                             |
| `sandbox`                     | [`SandboxSettings`](#sandboxsettings) ` \| None` | `None`              | Configure sandbox behavior programmatically. See [Sandbox settings](#sandboxsettings) for details                                        |
| `setting_sources`             | `list[SettingSource] \| None`                | `None` (no settings) | Control which filesystem settings to load. When omitted, no settings are loaded. **Note:** Must include `"project"` to load CLAUDE.md files                                             |
| `max_thinking_tokens`         | `int \| None`                                | `None`               | Maximum tokens for thinking blocks                                                                                                                                                      |

### `OutputFormat`

Configuration for structured output validation.

```python
class OutputFormat(TypedDict):
    type: Literal["json_schema"]
    schema: dict[str, Any]
```

| Field    | Required | Description                                    |
| :------- | :------- | :--------------------------------------------- |
| `type`   | Yes      | Must be `"json_schema"` for JSON Schema validation |
| `schema` | Yes      | JSON Schema definition for output validation   |

### `SystemPromptPreset`

Configuration for using Claude Code's preset system prompt with optional additions.

```python
class SystemPromptPreset(TypedDict):
    type: Literal["preset"]
    preset: Literal["claude_code"]
    append: NotRequired[str]
```

| Field    | Required | Description                                                   |
| :------- | :------- | :------------------------------------------------------------ |
| `type`   | Yes      | Must be `"preset"` to use a preset system prompt              |
| `preset` | Yes      | Must be `"claude_code"` to use Claude Code's system prompt    |
| `append` | No       | Additional instructions to append to the preset system prompt |

### `SettingSource`

Controls which filesystem-based configuration sources the SDK loads settings from.

```python
SettingSource = Literal["user", "project", "local"]
```

| Value       | Description                                  | Location                      |
| :---------- | :------------------------------------------- | :---------------------------- |
| `"user"`    | Global user settings                         | `~/.claude/settings.json`     |
| `"project"` | Shared project settings (version controlled) | `.claude/settings.json`       |
| `"local"`   | Local project settings (gitignored)          | `.claude/settings.local.json` |

#### Default behavior

When `setting_sources` is **omitted** or **`None`**, the SDK does **not** load any filesystem settings. This provides isolation for SDK applications.

#### Why use setting_sources?

**Load all filesystem settings (legacy behavior):**

```python
# Load all settings like SDK v0.0.x did
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="Analyze this code",
    options=ClaudeAgentOptions(
        setting_sources=["user", "project", "local"]  # Load all settings
    )
):
    print(message)
```

**Load only specific setting sources:**

```python
# Load only project settings, ignore user and local
async for message in query(
    prompt="Run CI checks",
    options=ClaudeAgentOptions(
        setting_sources=["project"]  # Only .claude/settings.json
    )
):
    print(message)
```

**Testing and CI environments:**

```python
# Ensure consistent behavior in CI by excluding local settings
async for message in query(
    prompt="Run tests",
    options=ClaudeAgentOptions(
        setting_sources=["project"],  # Only team-shared settings
        permission_mode="bypassPermissions"
    )
):
    print(message)
```

**SDK-only applications:**

```python
# Define everything programmatically (default behavior)
# No filesystem dependencies - setting_sources defaults to None
async for message in query(
    prompt="Review this PR",
    options=ClaudeAgentOptions(
        # setting_sources=None is the default, no need to specify
        agents={ /* ... */ },
        mcp_servers={ /* ... */ },
        allowed_tools=["Read", "Grep", "Glob"]
    )
):
    print(message)
```

**Loading CLAUDE.md project instructions:**

```python
# Load project settings to include CLAUDE.md files
async for message in query(
    prompt="Add a new feature following project conventions",
    options=ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code"  # Use Claude Code's system prompt
        },
        setting_sources=["project"],  # Required to load CLAUDE.md from project
        allowed_tools=["Read", "Write", "Edit"]
    )
):
    print(message)
```

#### Settings precedence

When multiple sources are loaded, settings are merged with this precedence (highest to lowest):

1. Local settings (`.claude/settings.local.json`)
2. Project settings (`.claude/settings.json`)
3. User settings (`~/.claude/settings.json`)

Programmatic options (like `agents`, `allowed_tools`) always override filesystem settings.

### `AgentDefinition`

Configuration for a subagent defined programmatically.

```python
@dataclass
class AgentDefinition:
    description: str
    prompt: str
    tools: list[str] | None = None
    model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None
```

| Field         | Required | Description                                                    |
| :------------ | :------- | :------------------------------------------------------------- |
| `description` | Yes      | Natural language description of when to use this agent         |
| `tools`       | No       | Array of allowed tool names. If omitted, inherits all tools    |
| `prompt`      | Yes      | The agent's system prompt                                      |
| `model`       | No       | Model override for this agent. If omitted, uses the main model |

### `PermissionMode`

Permission modes for controlling tool execution.

```python
PermissionMode = Literal[
    "default",           # Standard permission behavior
    "acceptEdits",       # Auto-accept file edits
    "plan",              # Planning mode - no execution
    "bypassPermissions"  # Bypass all permission checks (use with caution)
]
```

### `CanUseTool`

Type alias for tool permission callback functions.

```python
CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    Awaitable[PermissionResult]
]
```

The callback receives:
- `tool_name`: Name of the tool being called
- `input_data`: The tool's input parameters
- `context`: A `ToolPermissionContext` with additional information

Returns a `PermissionResult` (either `PermissionResultAllow` or `PermissionResultDeny`).

### `ToolPermissionContext`

Context information passed to tool permission callbacks.

```python
@dataclass
class ToolPermissionContext:
    signal: Any | None = None  # Future: abort signal support
    suggestions: list[PermissionUpdate] = field(default_factory=list)
```

| Field | Type | Description |
|:------|:-----|:------------|
| `signal` | `Any \| None` | Reserved for future abort signal support |
| `suggestions` | `list[PermissionUpdate]` | Permission update suggestions from the CLI |

### `PermissionResult`

Union type for permission callback results.

```python
PermissionResult = PermissionResultAllow | PermissionResultDeny
```

### `PermissionResultAllow`

Result indicating the tool call should be allowed.

```python
@dataclass
class PermissionResultAllow:
    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None
    updated_permissions: list[PermissionUpdate] | None = None
```

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `behavior` | `Literal["allow"]` | `"allow"` | Must be "allow" |
| `updated_input` | `dict[str, Any] \| None` | `None` | Modified input to use instead of original |
| `updated_permissions` | `list[PermissionUpdate] \| None` | `None` | Permission updates to apply |

### `PermissionResultDeny`

Result indicating the tool call should be denied.

```python
@dataclass
class PermissionResultDeny:
    behavior: Literal["deny"] = "deny"
    message: str = ""
    interrupt: bool = False
```

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `behavior` | `Literal["deny"]` | `"deny"` | Must be "deny" |
| `message` | `str` | `""` | Message explaining why the tool was denied |
| `interrupt` | `bool` | `False` | Whether to interrupt the current execution |

### `PermissionUpdate`

Configuration for updating permissions programmatically.

```python
@dataclass
class PermissionUpdate:
    type: Literal[
        "addRules",
        "replaceRules",
        "removeRules",
        "setMode",
        "addDirectories",
        "removeDirectories",
    ]
    rules: list[PermissionRuleValue] | None = None
    behavior: Literal["allow", "deny", "ask"] | None = None
    mode: PermissionMode | None = None
    directories: list[str] | None = None
    destination: Literal["userSettings", "projectSettings", "localSettings", "session"] | None = None
```

| Field | Type | Description |
|:------|:-----|:------------|
| `type` | `Literal[...]` | The type of permission update operation |
| `rules` | `list[PermissionRuleValue] \| None` | Rules for add/replace/remove operations |
| `behavior` | `Literal["allow", "deny", "ask"] \| None` | Behavior for rule-based operations |
| `mode` | `PermissionMode \| None` | Mode for setMode operation |
| `directories` | `list[str] \| None` | Directories for add/remove directory operations |
| `destination` | `Literal[...] \| None` | Where to apply the permission update |

### `SdkBeta`

Literal type for SDK beta features.

```python
SdkBeta = Literal["context-1m-2025-08-07"]
```

Use with the `betas` field in `ClaudeAgentOptions` to enable beta features.

### `McpSdkServerConfig`

Configuration for SDK MCP servers created with `create_sdk_mcp_server()`.

```python
class McpSdkServerConfig(TypedDict):
    type: Literal["sdk"]
    name: str
    instance: Any  # MCP Server instance
```

### `McpServerConfig`

Union type for MCP server configurations.

```python
McpServerConfig = McpStdioServerConfig | McpSSEServerConfig | McpHttpServerConfig | McpSdkServerConfig
```

#### `McpStdioServerConfig`

```python
class McpStdioServerConfig(TypedDict):
    type: NotRequired[Literal["stdio"]]  # Optional for backwards compatibility
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]
```

#### `McpSSEServerConfig`

```python
class McpSSEServerConfig(TypedDict):
    type: Literal["sse"]
    url: str
    headers: NotRequired[dict[str, str]]
```

#### `McpHttpServerConfig`

```python
class McpHttpServerConfig(TypedDict):
    type: Literal["http"]
    url: str
    headers: NotRequired[dict[str, str]]
```

### `SdkPluginConfig`

Configuration for loading plugins in the SDK.

```python
class SdkPluginConfig(TypedDict):
    type: Literal["local"]
    path: str
```

| Field | Type | Description |
|:------|:-----|:------------|
| `type` | `Literal["local"]` | Must be `"local"` (only local plugins currently supported) |
| `path` | `str` | Absolute or relative path to the plugin directory |

**Example:**
```python
plugins=[
    {"type": "local", "path": "./my-plugin"},
    {"type": "local", "path": "/absolute/path/to/plugin"}
]
```

For complete information on creating and using plugins, see [Plugins](/docs/en/agent-sdk/plugins).

## Message Types

### `Message`

Union type of all possible messages.

```python
Message = UserMessage | AssistantMessage | SystemMessage | ResultMessage | StreamEvent
```

### `UserMessage`

User input message.

```python
@dataclass
class UserMessage:
    content: str | list[ContentBlock]
```

### `AssistantMessage`

Assistant response message with content blocks.

```python
@dataclass
class AssistantMessage:
    content: list[ContentBlock]
    model: str
```

### `SystemMessage`

System message with metadata.

```python
@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any]
```

### `ResultMessage`

Final result message with cost and usage information.

```python
@dataclass
class ResultMessage:
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    result: str | None = None
    structured_output: Any = None
```

### `StreamEvent`

Stream event for partial message updates during streaming. Only received when `include_partial_messages=True` in `ClaudeAgentOptions`.

```python
@dataclass
class StreamEvent:
    uuid: str
    session_id: str
    event: dict[str, Any]  # The raw Anthropic API stream event
    parent_tool_use_id: str | None = None
```

| Field | Type | Description |
|:------|:-----|:------------|
| `uuid` | `str` | Unique identifier for this event |
| `session_id` | `str` | Session identifier |
| `event` | `dict[str, Any]` | The raw Anthropic API stream event data |
| `parent_tool_use_id` | `str \| None` | Parent tool use ID if this event is from a subagent |

## Content Block Types

### `ContentBlock`

Union type of all content blocks.

```python
ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock
```

### `TextBlock`

Text content block.

```python
@dataclass
class TextBlock:
    text: str
```

### `ThinkingBlock`

Thinking content block (for models with thinking capability).

```python
@dataclass
class ThinkingBlock:
    thinking: str
    signature: str
```

### `ToolUseBlock`

Tool use request block.

```python
@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
```

### `ToolResultBlock`

Tool execution result block.

```python
@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None
```

## Error Types

### `ClaudeSDKError`

Base exception class for all SDK errors.

```python
class ClaudeSDKError(Exception):
    """Base error for Claude SDK."""
```

### `CLINotFoundError`

Raised when Claude Code CLI is not installed or not found.

```python
class CLINotFoundError(CLIConnectionError):
    def __init__(self, message: str = "Claude Code not found", cli_path: str | None = None):
        """
        Args:
            message: Error message (default: "Claude Code not found")
            cli_path: Optional path to the CLI that was not found
        """
```

### `CLIConnectionError`

Raised when connection to Claude Code fails.

```python
class CLIConnectionError(ClaudeSDKError):
    """Failed to connect to Claude Code."""
```

### `ProcessError`

Raised when the Claude Code process fails.

```python
class ProcessError(ClaudeSDKError):
    def __init__(self, message: str, exit_code: int | None = None, stderr: str | None = None):
        self.exit_code = exit_code
        self.stderr = stderr
```

### `CLIJSONDecodeError`

Raised when JSON parsing fails.

```python
class CLIJSONDecodeError(ClaudeSDKError):
    def __init__(self, line: str, original_error: Exception):
        """
        Args:
            line: The line that failed to parse
            original_error: The original JSON decode exception
        """
        self.line = line
        self.original_error = original_error
```

## Hook Types

For a comprehensive guide on using hooks with examples and common patterns, see the [Hooks guide](/docs/en/agent-sdk/hooks).

### `HookEvent`

Supported hook event types. Note that due to setup limitations, the Python SDK does not support SessionStart, SessionEnd, and Notification hooks.

```python
HookEvent = Literal[
    "PreToolUse",      # Called before tool execution
    "PostToolUse",     # Called after tool execution
    "UserPromptSubmit", # Called when user submits a prompt
    "Stop",            # Called when stopping execution
    "SubagentStop",    # Called when a subagent stops
    "PreCompact"       # Called before message compaction
]
```

### `HookCallback`

Type definition for hook callback functions.

```python
HookCallback = Callable[
    [dict[str, Any], str | None, HookContext],
    Awaitable[dict[str, Any]]
]
```

Parameters:

- `input_data`: Hook-specific input data (see [Hooks guide](/docs/en/agent-sdk/hooks#input-data))
- `tool_use_id`: Optional tool use identifier (for tool-related hooks)
- `context`: Hook context with additional information

Returns a dictionary that may contain:

- `decision`: `"block"` to block the action
- `systemMessage`: System message to add to the transcript
- `hookSpecificOutput`: Hook-specific output data

### `HookContext`

Context information passed to hook callbacks.

```python
@dataclass
class HookContext:
    signal: Any | None = None  # Future: abort signal support
```

### `HookMatcher`

Configuration for matching hooks to specific events or tools.

```python
@dataclass
class HookMatcher:
    matcher: str | None = None        # Tool name or pattern to match (e.g., "Bash", "Write|Edit")
    hooks: list[HookCallback] = field(default_factory=list)  # List of callbacks to execute
    timeout: float | None = None        # Timeout in seconds for all hooks in this matcher (default: 60)
```

### `HookInput`

Union type of all hook input types. The actual type depends on the `hook_event_name` field.

```python
HookInput = (
    PreToolUseHookInput
    | PostToolUseHookInput
    | UserPromptSubmitHookInput
    | StopHookInput
    | SubagentStopHookInput
    | PreCompactHookInput
)
```

### `BaseHookInput`

Base fields present in all hook input types.

```python
class BaseHookInput(TypedDict):
    session_id: str
    transcript_path: str
    cwd: str
    permission_mode: NotRequired[str]
```

| Field | Type | Description |
|:------|:-----|:------------|
| `session_id` | `str` | Current session identifier |
| `transcript_path` | `str` | Path to the session transcript file |
| `cwd` | `str` | Current working directory |
| `permission_mode` | `str` (optional) | Current permission mode |

### `PreToolUseHookInput`

Input data for `PreToolUse` hook events.

```python
class PreToolUseHookInput(BaseHookInput):
    hook_event_name: Literal["PreToolUse"]
    tool_name: str
    tool_input: dict[str, Any]
```

| Field | Type | Description |
|:------|:-----|:------------|
| `hook_event_name` | `Literal["PreToolUse"]` | Always "PreToolUse" |
| `tool_name` | `str` | Name of the tool about to be executed |
| `tool_input` | `dict[str, Any]` | Input parameters for the tool |

### `PostToolUseHookInput`

Input data for `PostToolUse` hook events.

```python
class PostToolUseHookInput(BaseHookInput):
    hook_event_name: Literal["PostToolUse"]
    tool_name: str
    tool_input: dict[str, Any]
    tool_response: Any
```

| Field | Type | Description |
|:------|:-----|:------------|
| `hook_event_name` | `Literal["PostToolUse"]` | Always "PostToolUse" |
| `tool_name` | `str` | Name of the tool that was executed |
| `tool_input` | `dict[str, Any]` | Input parameters that were used |
| `tool_response` | `Any` | Response from the tool execution |

### `UserPromptSubmitHookInput`

Input data for `UserPromptSubmit` hook events.

```python
class UserPromptSubmitHookInput(BaseHookInput):
    hook_event_name: Literal["UserPromptSubmit"]
    prompt: str
```

| Field | Type | Description |
|:------|:-----|:------------|
| `hook_event_name` | `Literal["UserPromptSubmit"]` | Always "UserPromptSubmit" |
| `prompt` | `str` | The user's submitted prompt |

### `StopHookInput`

Input data for `Stop` hook events.

```python
class StopHookInput(BaseHookInput):
    hook_event_name: Literal["Stop"]
    stop_hook_active: bool
```

| Field | Type | Description |
|:------|:-----|:------------|
| `hook_event_name` | `Literal["Stop"]` | Always "Stop" |
| `stop_hook_active` | `bool` | Whether the stop hook is active |

### `SubagentStopHookInput`

Input data for `SubagentStop` hook events.

```python
class SubagentStopHookInput(BaseHookInput):
    hook_event_name: Literal["SubagentStop"]
    stop_hook_active: bool
```

| Field | Type | Description |
|:------|:-----|:------------|
| `hook_event_name` | `Literal["SubagentStop"]` | Always "SubagentStop" |
| `stop_hook_active` | `bool` | Whether the stop hook is active |

### `PreCompactHookInput`

Input data for `PreCompact` hook events.

```python
class PreCompactHookInput(BaseHookInput):
    hook_event_name: Literal["PreCompact"]
    trigger: Literal["manual", "auto"]
    custom_instructions: str | None
```

| Field | Type | Description |
|:------|:-----|:------------|
| `hook_event_name` | `Literal["PreCompact"]` | Always "PreCompact" |
| `trigger` | `Literal["manual", "auto"]` | What triggered the compaction |
| `custom_instructions` | `str \| None` | Custom instructions for compaction |

### `HookJSONOutput`

Union type for hook callback return values.

```python
HookJSONOutput = AsyncHookJSONOutput | SyncHookJSONOutput
```

#### `SyncHookJSONOutput`

Synchronous hook output with control and decision fields.

```python
class SyncHookJSONOutput(TypedDict):
    # Control fields
    continue_: NotRequired[bool]      # Whether to proceed (default: True)
    suppressOutput: NotRequired[bool] # Hide stdout from transcript
    stopReason: NotRequired[str]      # Message when continue is False

    # Decision fields
    decision: NotRequired[Literal["block"]]
    systemMessage: NotRequired[str]   # Warning message for user
    reason: NotRequired[str]          # Feedback for Claude

    # Hook-specific output
    hookSpecificOutput: NotRequired[dict[str, Any]]
```

<Note>
Use `continue_` (with underscore) in Python code. It is automatically converted to `continue` when sent to the CLI.
</Note>

#### `AsyncHookJSONOutput`

Async hook output that defers hook execution.

```python
class AsyncHookJSONOutput(TypedDict):
    async_: Literal[True]             # Set to True to defer execution
    asyncTimeout: NotRequired[int]    # Timeout in milliseconds
```

<Note>
Use `async_` (with underscore) in Python code. It is automatically converted to `async` when sent to the CLI.
</Note>

### Hook Usage Example

This example registers two hooks: one that blocks dangerous bash commands like `rm -rf /`, and another that logs all tool usage for auditing. The security hook only runs on Bash commands (via the `matcher`), while the logging hook runs on all tools.

```python
from claude_agent_sdk import query, ClaudeAgentOptions, HookMatcher, HookContext
from typing import Any

async def validate_bash_command(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext
) -> dict[str, Any]:
    """Validate and potentially block dangerous bash commands."""
    if input_data['tool_name'] == 'Bash':
        command = input_data['tool_input'].get('command', '')
        if 'rm -rf /' in command:
            return {
                'hookSpecificOutput': {
                    'hookEventName': 'PreToolUse',
                    'permissionDecision': 'deny',
                    'permissionDecisionReason': 'Dangerous command blocked'
                }
            }
    return {}

async def log_tool_use(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext
) -> dict[str, Any]:
    """Log all tool usage for auditing."""
    print(f"Tool used: {input_data.get('tool_name')}")
    return {}

options = ClaudeAgentOptions(
    hooks={
        'PreToolUse': [
            HookMatcher(matcher='Bash', hooks=[validate_bash_command], timeout=120),  # 2 min for validation
            HookMatcher(hooks=[log_tool_use])  # Applies to all tools (default 60s timeout)
        ],
        'PostToolUse': [
            HookMatcher(hooks=[log_tool_use])
        ]
    }
)

async for message in query(
    prompt="Analyze this codebase",
    options=options
):
    print(message)
```

## Tool Input/Output Types

Documentation of input/output schemas for all built-in Claude Code tools. While the Python SDK doesn't export these as types, they represent the structure of tool inputs and outputs in messages.

### Task

**Tool name:** `Task`

**Input:**

```python
{
    "description": str,      # A short (3-5 word) description of the task
    "prompt": str,           # The task for the agent to perform
    "subagent_type": str     # The type of specialized agent to use
}
```

**Output:**

```python
{
    "result": str,                    # Final result from the subagent
    "usage": dict | None,             # Token usage statistics
    "total_cost_usd": float | None,  # Total cost in USD
    "duration_ms": int | None         # Execution duration in milliseconds
}
```

### AskUserQuestion

**Tool name:** `AskUserQuestion`

Asks the user clarifying questions during execution. See [Handle approvals and user input](/docs/en/agent-sdk/user-input#handle-clarifying-questions) for usage details.

**Input:**

```python
{
    "questions": [                    # Questions to ask the user (1-4 questions)
        {
            "question": str,          # The complete question to ask the user
            "header": str,            # Very short label displayed as a chip/tag (max 12 chars)
            "options": [              # The available choices (2-4 options)
                {
                    "label": str,         # Display text for this option (1-5 words)
                    "description": str    # Explanation of what this option means
                }
            ],
            "multiSelect": bool       # Set to true to allow multiple selections
        }
    ],
    "answers": dict | None            # User answers populated by the permission system
}
```

**Output:**

```python
{
    "questions": [                    # The questions that were asked
        {
            "question": str,
            "header": str,
            "options": [{"label": str, "description": str}],
            "multiSelect": bool
        }
    ],
    "answers": dict[str, str]         # Maps question text to answer string
                                      # Multi-select answers are comma-separated
}
```

### Bash

**Tool name:** `Bash`

**Input:**

```python
{
    "command": str,                  # The command to execute
    "timeout": int | None,           # Optional timeout in milliseconds (max 600000)
    "description": str | None,       # Clear, concise description (5-10 words)
    "run_in_background": bool | None # Set to true to run in background
}
```

**Output:**

```python
{
    "output": str,              # Combined stdout and stderr output
    "exitCode": int,            # Exit code of the command
    "killed": bool | None,      # Whether command was killed due to timeout
    "shellId": str | None       # Shell ID for background processes
}
```

### Edit

**Tool name:** `Edit`

**Input:**

```python
{
    "file_path": str,           # The absolute path to the file to modify
    "old_string": str,          # The text to replace
    "new_string": str,          # The text to replace it with
    "replace_all": bool | None  # Replace all occurrences (default False)
}
```

**Output:**

```python
{
    "message": str,      # Confirmation message
    "replacements": int, # Number of replacements made
    "file_path": str     # File path that was edited
}
```

### Read

**Tool name:** `Read`

**Input:**

```python
{
    "file_path": str,       # The absolute path to the file to read
    "offset": int | None,   # The line number to start reading from
    "limit": int | None     # The number of lines to read
}
```

**Output (Text files):**

```python
{
    "content": str,         # File contents with line numbers
    "total_lines": int,     # Total number of lines in file
    "lines_returned": int   # Lines actually returned
}
```

**Output (Images):**

```python
{
    "image": str,       # Base64 encoded image data
    "mime_type": str,   # Image MIME type
    "file_size": int    # File size in bytes
}
```

### Write

**Tool name:** `Write`

**Input:**

```python
{
    "file_path": str,  # The absolute path to the file to write
    "content": str     # The content to write to the file
}
```

**Output:**

```python
{
    "message": str,        # Success message
    "bytes_written": int,  # Number of bytes written
    "file_path": str       # File path that was written
}
```

### Glob

**Tool name:** `Glob`

**Input:**

```python
{
    "pattern": str,       # The glob pattern to match files against
    "path": str | None    # The directory to search in (defaults to cwd)
}
```

**Output:**

```python
{
    "matches": list[str],  # Array of matching file paths
    "count": int,          # Number of matches found
    "search_path": str     # Search directory used
}
```

### Grep

**Tool name:** `Grep`

**Input:**

```python
{
    "pattern": str,                    # The regular expression pattern
    "path": str | None,                # File or directory to search in
    "glob": str | None,                # Glob pattern to filter files
    "type": str | None,                # File type to search
    "output_mode": str | None,         # "content", "files_with_matches", or "count"
    "-i": bool | None,                 # Case insensitive search
    "-n": bool | None,                 # Show line numbers
    "-B": int | None,                  # Lines to show before each match
    "-A": int | None,                  # Lines to show after each match
    "-C": int | None,                  # Lines to show before and after
    "head_limit": int | None,          # Limit output to first N lines/entries
    "multiline": bool | None           # Enable multiline mode
}
```

**Output (content mode):**

```python
{
    "matches": [
        {
            "file": str,
            "line_number": int | None,
            "line": str,
            "before_context": list[str] | None,
            "after_context": list[str] | None
        }
    ],
    "total_matches": int
}
```

**Output (files_with_matches mode):**

```python
{
    "files": list[str],  # Files containing matches
    "count": int         # Number of files with matches
}
```

### NotebookEdit

**Tool name:** `NotebookEdit`

**Input:**

```python
{
    "notebook_path": str,                     # Absolute path to the Jupyter notebook
    "cell_id": str | None,                    # The ID of the cell to edit
    "new_source": str,                        # The new source for the cell
    "cell_type": "code" | "markdown" | None,  # The type of the cell
    "edit_mode": "replace" | "insert" | "delete" | None  # Edit operation type
}
```

**Output:**

```python
{
    "message": str,                              # Success message
    "edit_type": "replaced" | "inserted" | "deleted",  # Type of edit performed
    "cell_id": str | None,                       # Cell ID that was affected
    "total_cells": int                           # Total cells in notebook after edit
}
```

### WebFetch

**Tool name:** `WebFetch`

**Input:**

```python
{
    "url": str,     # The URL to fetch content from
    "prompt": str   # The prompt to run on the fetched content
}
```

**Output:**

```python
{
    "response": str,           # AI model's response to the prompt
    "url": str,                # URL that was fetched
    "final_url": str | None,   # Final URL after redirects
    "status_code": int | None  # HTTP status code
}
```

### WebSearch

**Tool name:** `WebSearch`

**Input:**

```python
{
    "query": str,                        # The search query to use
    "allowed_domains": list[str] | None, # Only include results from these domains
    "blocked_domains": list[str] | None  # Never include results from these domains
}
```

**Output:**

```python
{
    "results": [
        {
            "title": str,
            "url": str,
            "snippet": str,
            "metadata": dict | None
        }
    ],
    "total_results": int,
    "query": str
}
```

### TodoWrite

**Tool name:** `TodoWrite`

**Input:**

```python
{
    "todos": [
        {
            "content": str,                              # The task description
            "status": "pending" | "in_progress" | "completed",  # Task status
            "activeForm": str                            # Active form of the description
        }
    ]
}
```

**Output:**

```python
{
    "message": str,  # Success message
    "stats": {
        "total": int,
        "pending": int,
        "in_progress": int,
        "completed": int
    }
}
```

### BashOutput

**Tool name:** `BashOutput`

**Input:**

```python
{
    "bash_id": str,       # The ID of the background shell
    "filter": str | None  # Optional regex to filter output lines
}
```

**Output:**

```python
{
    "output": str,                                      # New output since last check
    "status": "running" | "completed" | "failed",       # Current shell status
    "exitCode": int | None                              # Exit code when completed
}
```

### KillBash

**Tool name:** `KillBash`

**Input:**

```python
{
    "shell_id": str  # The ID of the background shell to kill
}
```

**Output:**

```python
{
    "message": str,  # Success message
    "shell_id": str  # ID of the killed shell
}
```

### ExitPlanMode

**Tool name:** `ExitPlanMode`

**Input:**

```python
{
    "plan": str  # The plan to run by the user for approval
}
```

**Output:**

```python
{
    "message": str,          # Confirmation message
    "approved": bool | None  # Whether user approved the plan
}
```

### ListMcpResources

**Tool name:** `ListMcpResources`

**Input:**

```python
{
    "server": str | None  # Optional server name to filter resources by
}
```

**Output:**

```python
{
    "resources": [
        {
            "uri": str,
            "name": str,
            "description": str | None,
            "mimeType": str | None,
            "server": str
        }
    ],
    "total": int
}
```

### ReadMcpResource

**Tool name:** `ReadMcpResource`

**Input:**

```python
{
    "server": str,  # The MCP server name
    "uri": str      # The resource URI to read
}
```

**Output:**

```python
{
    "contents": [
        {
            "uri": str,
            "mimeType": str | None,
            "text": str | None,
            "blob": str | None
        }
    ],
    "server": str
}
```

## Advanced Features with ClaudeSDKClient

### Building a Continuous Conversation Interface

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock
import asyncio

class ConversationSession:
    """Maintains a single conversation session with Claude."""

    def __init__(self, options: ClaudeAgentOptions = None):
        self.client = ClaudeSDKClient(options)
        self.turn_count = 0

    async def start(self):
        await self.client.connect()
        print("Starting conversation session. Claude will remember context.")
        print("Commands: 'exit' to quit, 'interrupt' to stop current task, 'new' for new session")

        while True:
            user_input = input(f"\n[Turn {self.turn_count + 1}] You: ")

            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'interrupt':
                await self.client.interrupt()
                print("Task interrupted!")
                continue
            elif user_input.lower() == 'new':
                # Disconnect and reconnect for a fresh session
                await self.client.disconnect()
                await self.client.connect()
                self.turn_count = 0
                print("Started new conversation session (previous context cleared)")
                continue

            # Send message - Claude remembers all previous messages in this session
            await self.client.query(user_input)
            self.turn_count += 1

            # Process response
            print(f"[Turn {self.turn_count}] Claude: ", end="")
            async for message in self.client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text, end="")
            print()  # New line after response

        await self.client.disconnect()
        print(f"Conversation ended after {self.turn_count} turns.")

async def main():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash"],
        permission_mode="acceptEdits"
    )
    session = ConversationSession(options)
    await session.start()

# Example conversation:
# Turn 1 - You: "Create a file called hello.py"
# Turn 1 - Claude: "I'll create a hello.py file for you..."
# Turn 2 - You: "What's in that file?"
# Turn 2 - Claude: "The hello.py file I just created contains..." (remembers!)
# Turn 3 - You: "Add a main function to it"
# Turn 3 - Claude: "I'll add a main function to hello.py..." (knows which file!)

asyncio.run(main())
```

### Using Hooks for Behavior Modification

```python
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    HookMatcher,
    HookContext
)
import asyncio
from typing import Any

async def pre_tool_logger(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext
) -> dict[str, Any]:
    """Log all tool usage before execution."""
    tool_name = input_data.get('tool_name', 'unknown')
    print(f"[PRE-TOOL] About to use: {tool_name}")

    # You can modify or block the tool execution here
    if tool_name == "Bash" and "rm -rf" in str(input_data.get('tool_input', {})):
        return {
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'deny',
                'permissionDecisionReason': 'Dangerous command blocked'
            }
        }
    return {}

async def post_tool_logger(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext
) -> dict[str, Any]:
    """Log results after tool execution."""
    tool_name = input_data.get('tool_name', 'unknown')
    print(f"[POST-TOOL] Completed: {tool_name}")
    return {}

async def user_prompt_modifier(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext
) -> dict[str, Any]:
    """Add context to user prompts."""
    original_prompt = input_data.get('prompt', '')

    # Add timestamp to all prompts
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        'hookSpecificOutput': {
            'hookEventName': 'UserPromptSubmit',
            'updatedPrompt': f"[{timestamp}] {original_prompt}"
        }
    }

async def main():
    options = ClaudeAgentOptions(
        hooks={
            'PreToolUse': [
                HookMatcher(hooks=[pre_tool_logger]),
                HookMatcher(matcher='Bash', hooks=[pre_tool_logger])
            ],
            'PostToolUse': [
                HookMatcher(hooks=[post_tool_logger])
            ],
            'UserPromptSubmit': [
                HookMatcher(hooks=[user_prompt_modifier])
            ]
        },
        allowed_tools=["Read", "Write", "Bash"]
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("List files in current directory")

        async for message in client.receive_response():
            # Hooks will automatically log tool usage
            pass

asyncio.run(main())
```

### Real-time Progress Monitoring

```python
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ToolUseBlock,
    ToolResultBlock,
    TextBlock
)
import asyncio

async def monitor_progress():
    options = ClaudeAgentOptions(
        allowed_tools=["Write", "Bash"],
        permission_mode="acceptEdits"
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "Create 5 Python files with different sorting algorithms"
        )

        # Monitor progress in real-time
        files_created = []
        async for message in client.receive_messages():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        if block.name == "Write":
                            file_path = block.input.get("file_path", "")
                            print(f"🔨 Creating: {file_path}")
                    elif isinstance(block, ToolResultBlock):
                        print(f"✅ Completed tool execution")
                    elif isinstance(block, TextBlock):
                        print(f"💭 Claude says: {block.text[:100]}...")

            # Check if we've received the final result
            if hasattr(message, 'subtype') and message.subtype in ['success', 'error']:
                print(f"\n🎯 Task completed!")
                break

asyncio.run(monitor_progress())
```

## Example Usage

### Basic file operations (using query)

```python
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ToolUseBlock
import asyncio

async def create_project():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash"],
        permission_mode='acceptEdits',
        cwd="/home/user/project"
    )

    async for message in query(
        prompt="Create a Python project structure with setup.py",
        options=options
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"Using tool: {block.name}")

asyncio.run(create_project())
```

### Error handling

```python
from claude_agent_sdk import (
    query,
    CLINotFoundError,
    ProcessError,
    CLIJSONDecodeError
)

try:
    async for message in query(prompt="Hello"):
        print(message)
except CLINotFoundError:
    print("Please install Claude Code: npm install -g @anthropic-ai/claude-code")
except ProcessError as e:
    print(f"Process failed with exit code: {e.exit_code}")
except CLIJSONDecodeError as e:
    print(f"Failed to parse response: {e}")
```

### Streaming mode with client

```python
from claude_agent_sdk import ClaudeSDKClient
import asyncio

async def interactive_session():
    async with ClaudeSDKClient() as client:
        # Send initial message
        await client.query("What's the weather like?")

        # Process responses
        async for msg in client.receive_response():
            print(msg)

        # Send follow-up
        await client.query("Tell me more about that")

        # Process follow-up response
        async for msg in client.receive_response():
            print(msg)

asyncio.run(interactive_session())
```

### Using custom tools with ClaudeSDKClient

```python
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
    AssistantMessage,
    TextBlock
)
import asyncio
from typing import Any

# Define custom tools with @tool decorator
@tool("calculate", "Perform mathematical calculations", {"expression": str})
async def calculate(args: dict[str, Any]) -> dict[str, Any]:
    try:
        result = eval(args["expression"], {"__builtins__": {}})
        return {
            "content": [{
                "type": "text",
                "text": f"Result: {result}"
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error: {str(e)}"
            }],
            "is_error": True
        }

@tool("get_time", "Get current time", {})
async def get_time(args: dict[str, Any]) -> dict[str, Any]:
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "content": [{
            "type": "text",
            "text": f"Current time: {current_time}"
        }]
    }

async def main():
    # Create SDK MCP server with custom tools
    my_server = create_sdk_mcp_server(
        name="utilities",
        version="1.0.0",
        tools=[calculate, get_time]
    )

    # Configure options with the server
    options = ClaudeAgentOptions(
        mcp_servers={"utils": my_server},
        allowed_tools=[
            "mcp__utils__calculate",
            "mcp__utils__get_time"
        ]
    )

    # Use ClaudeSDKClient for interactive tool usage
    async with ClaudeSDKClient(options=options) as client:
        await client.query("What's 123 * 456?")

        # Process calculation response
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Calculation: {block.text}")

        # Follow up with time query
        await client.query("What time is it now?")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Time: {block.text}")

asyncio.run(main())
```

## Sandbox Configuration

### `SandboxSettings`

Configuration for sandbox behavior. Use this to enable command sandboxing and configure network restrictions programmatically.

```python
class SandboxSettings(TypedDict, total=False):
    enabled: bool
    autoAllowBashIfSandboxed: bool
    excludedCommands: list[str]
    allowUnsandboxedCommands: bool
    network: SandboxNetworkConfig
    ignoreViolations: SandboxIgnoreViolations
    enableWeakerNestedSandbox: bool
```

| Property | Type | Default | Description |
| :------- | :--- | :------ | :---------- |
| `enabled` | `bool` | `False` | Enable sandbox mode for command execution |
| `autoAllowBashIfSandboxed` | `bool` | `False` | Auto-approve bash commands when sandbox is enabled |
| `excludedCommands` | `list[str]` | `[]` | Commands that always bypass sandbox restrictions (e.g., `["docker"]`). These run unsandboxed automatically without model involvement |
| `allowUnsandboxedCommands` | `bool` | `False` | Allow the model to request running commands outside the sandbox. When `True`, the model can set `dangerouslyDisableSandbox` in tool input, which falls back to the [permissions system](#permissions-fallback-for-unsandboxed-commands) |
| `network` | [`SandboxNetworkConfig`](#sandboxnetworkconfig) | `None` | Network-specific sandbox configuration |
| `ignoreViolations` | [`SandboxIgnoreViolations`](#sandboxignoreviolations) | `None` | Configure which sandbox violations to ignore |
| `enableWeakerNestedSandbox` | `bool` | `False` | Enable a weaker nested sandbox for compatibility |

<Note>
**Filesystem and network access restrictions** are NOT configured via sandbox settings. Instead, they are derived from [permission rules](https://code.claude.com/docs/en/settings#permission-settings):

- **Filesystem read restrictions**: Read deny rules
- **Filesystem write restrictions**: Edit allow/deny rules
- **Network restrictions**: WebFetch allow/deny rules

Use sandbox settings for command execution sandboxing, and permission rules for filesystem and network access control.
</Note>

#### Example usage

```python
from claude_agent_sdk import query, ClaudeAgentOptions, SandboxSettings

sandbox_settings: SandboxSettings = {
    "enabled": True,
    "autoAllowBashIfSandboxed": True,
    "network": {
        "allowLocalBinding": True
    }
}

async for message in query(
    prompt="Build and test my project",
    options=ClaudeAgentOptions(sandbox=sandbox_settings)
):
    print(message)
```

<Warning>
**Unix socket security**: The `allowUnixSockets` option can grant access to powerful system services. For example, allowing `/var/run/docker.sock` effectively grants full host system access through the Docker API, bypassing sandbox isolation. Only allow Unix sockets that are strictly necessary and understand the security implications of each.
</Warning>

### `SandboxNetworkConfig`

Network-specific configuration for sandbox mode.

```python
class SandboxNetworkConfig(TypedDict, total=False):
    allowLocalBinding: bool
    allowUnixSockets: list[str]
    allowAllUnixSockets: bool
    httpProxyPort: int
    socksProxyPort: int
```

| Property | Type | Default | Description |
| :------- | :--- | :------ | :---------- |
| `allowLocalBinding` | `bool` | `False` | Allow processes to bind to local ports (e.g., for dev servers) |
| `allowUnixSockets` | `list[str]` | `[]` | Unix socket paths that processes can access (e.g., Docker socket) |
| `allowAllUnixSockets` | `bool` | `False` | Allow access to all Unix sockets |
| `httpProxyPort` | `int` | `None` | HTTP proxy port for network requests |
| `socksProxyPort` | `int` | `None` | SOCKS proxy port for network requests |

### `SandboxIgnoreViolations`

Configuration for ignoring specific sandbox violations.

```python
class SandboxIgnoreViolations(TypedDict, total=False):
    file: list[str]
    network: list[str]
```

| Property | Type | Default | Description |
| :------- | :--- | :------ | :---------- |
| `file` | `list[str]` | `[]` | File path patterns to ignore violations for |
| `network` | `list[str]` | `[]` | Network patterns to ignore violations for |

### Permissions Fallback for Unsandboxed Commands

When `allowUnsandboxedCommands` is enabled, the model can request to run commands outside the sandbox by setting `dangerouslyDisableSandbox: True` in the tool input. These requests fall back to the existing permissions system, meaning your `can_use_tool` handler will be invoked, allowing you to implement custom authorization logic.

<Note>
**`excludedCommands` vs `allowUnsandboxedCommands`:**
- `excludedCommands`: A static list of commands that always bypass the sandbox automatically (e.g., `["docker"]`). The model has no control over this.
- `allowUnsandboxedCommands`: Lets the model decide at runtime whether to request unsandboxed execution by setting `dangerouslyDisableSandbox: True` in the tool input.
</Note>

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async def can_use_tool(tool: str, input: dict) -> bool:
    # Check if the model is requesting to bypass the sandbox
    if tool == "Bash" and input.get("dangerouslyDisableSandbox"):
        # The model wants to run this command outside the sandbox
        print(f"Unsandboxed command requested: {input.get('command')}")

        # Return True to allow, False to deny
        return is_command_authorized(input.get("command"))
    return True

async def main():
    async for message in query(
        prompt="Deploy my application",
        options=ClaudeAgentOptions(
            sandbox={
                "enabled": True,
                "allowUnsandboxedCommands": True  # Model can request unsandboxed execution
            },
            permission_mode="default",
            can_use_tool=can_use_tool
        )
    ):
        print(message)
```

This pattern enables you to:

- **Audit model requests**: Log when the model requests unsandboxed execution
- **Implement allowlists**: Only permit specific commands to run unsandboxed
- **Add approval workflows**: Require explicit authorization for privileged operations

<Warning>
Commands running with `dangerouslyDisableSandbox: True` have full system access. Ensure your `can_use_tool` handler validates these requests carefully.

If `permission_mode` is set to `bypassPermissions` and `allow_unsandboxed_commands` is enabled, the model can autonomously execute commands outside the sandbox without any approval prompts. This combination effectively allows the model to escape sandbox isolation silently.
</Warning>

## See also

- [Python SDK guide](/docs/en/agent-sdk/python) - Tutorial and examples
- [SDK overview](/docs/en/agent-sdk/overview) - General SDK concepts
- [TypeScript SDK reference](/docs/en/agent-sdk/typescript) - TypeScript SDK documentation
- [CLI reference](https://code.claude.com/docs/en/cli-reference) - Command-line interface
- [Common workflows](https://code.claude.com/docs/en/common-workflows) - Step-by-step guides
