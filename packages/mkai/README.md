# mkai

Alias package for [markitai](https://pypi.org/project/markitai/) — an opinionated
Markdown converter with native LLM enhancement support.

This package contains no code: it exists so that `pip install mkai` /
`uv tool install mkai` works and to protect the `mkai` name (the short command
both packages install). All functionality lives in
[markitai](https://github.com/Ynewtime/markitai), which this package depends on.

```bash
uv tool install "markitai[all]"   # canonical
# or
uv tool install "mkai[all]"       # same thing via this alias
```
