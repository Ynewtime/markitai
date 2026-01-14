## Browser Automation

> source: https://github.com/vercel-labs/agent-browser

Run `agent-browser --help` for all commands.

Use `agent-browser` for web automation.

Core workflow:
1. `agent-browser open <url>` - Navigate to page
2. `agent-browser snapshot -i` - Get interactive elements with refs (@e1, @e2)
3. `agent-browser click @e1` / `fill @e2 "text"` - Interact using refs
4. Re-snapshot after page changes
