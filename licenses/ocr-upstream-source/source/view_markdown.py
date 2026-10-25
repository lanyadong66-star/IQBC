"""Render WeVisDoc Markdown output in a browser.

Edit ``MARKDOWN_PATH`` below and run this file directly from PyCharm.
The script writes a sibling ``.preview.html`` file and opens it in the
default browser.
"""

import html
from pathlib import Path
import sys
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# 在这里填写要查看的 Markdown 文件或目录，然后直接运行本文件。
# 示例：
# MARKDOWN_PATH = r"/home/tq/桌面/ceshi/WeVisDoc_json/23.png.md"
# MARKDOWN_PATH = r"D:\\Desktop\\WeVisDoc-main\\outputs\\local_predictions"
# 如果填写目录，脚本会自动选择其中最近修改的 .md 文件。
# ============================================================
MARKDOWN_PATH = r""

# 留空时，HTML 文件会生成在 Markdown 文件旁边。
HTML_OUTPUT_PATH = r""

# 是否生成 HTML 后自动打开浏览器。
OPEN_BROWSER = True

STYLE = """
:root {
    color-scheme: light;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
                 "Microsoft YaHei", Arial, sans-serif;
    line-height: 1.7;
}
body {
    max-width: 1100px;
    margin: 0 auto;
    padding: 36px 28px 64px;
    color: #202124;
    background: #ffffff;
}
h1, h2, h3, h4, h5, h6 {
    line-height: 1.3;
    margin-top: 1.6em;
}
h1, h2 {
    border-bottom: 1px solid #dadce0;
    padding-bottom: 0.3em;
}
p, ul, ol, blockquote, pre, table {
    margin: 1em 0;
}
blockquote {
    margin-left: 0;
    padding: 0.2em 1em;
    color: #5f6368;
    border-left: 4px solid #dadce0;
}
table {
    border-collapse: collapse;
    display: block;
    max-width: 100%;
    overflow-x: auto;
}
th, td {
    border: 1px solid #b7b7b7;
    padding: 7px 12px;
    text-align: left;
    vertical-align: top;
}
th {
    background: #f1f3f4;
}
code {
    padding: 0.12em 0.35em;
    border-radius: 4px;
    background: #f1f3f4;
}
pre {
    padding: 14px 16px;
    overflow-x: auto;
    border-radius: 6px;
    background: #f6f8fa;
}
pre code {
    padding: 0;
    background: transparent;
}
img {
    max-width: 100%;
    height: auto;
}
.source-name {
    color: #5f6368;
    font-size: 0.9rem;
    margin-bottom: 2em;
}
"""


def find_markdown(path: Path) -> Path:
    """Resolve a file or choose the newest Markdown file in a directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError(f"输入文件不是 Markdown 文件: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"找不到文件或目录: {path}")

    candidates = sorted(
        (item for item in path.iterdir()
         if item.is_file() and item.suffix.lower() in {".md", ".markdown"}),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"目录中没有 Markdown 文件: {path}")
    return candidates[0]


def render_markdown(md_path: Path, html_path: Path) -> None:
    """Convert Markdown to a self-contained HTML preview page."""
    try:
        import markdown
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Markdown 依赖，请先运行: python -m pip install Markdown"
        ) from exc

    source = md_path.read_text(encoding="utf-8-sig")
    body = markdown.markdown(
        source,
        extensions=[
            "extra",       # tables, fenced code, footnotes, definition lists
            "toc",         # heading anchors
            "sane_lists",
            "nl2br",        # preserve line breaks from OCR output
        ],
        output_format="html5",
    )
    title = html.escape(md_path.name)
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{STYLE}</style>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']]
  }}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
<div class="source-name">来源: {title}</div>
{body}
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")


def main() -> int:
    if not MARKDOWN_PATH.strip():
        print(
            "请先在 view_markdown.py 顶部的 MARKDOWN_PATH = r\"...\" 中填写 Markdown 路径。",
            file=sys.stderr,
        )
        return 1
    try:
        md_path = find_markdown(Path(MARKDOWN_PATH))
        html_path = (
            Path(HTML_OUTPUT_PATH).expanduser().resolve()
            if HTML_OUTPUT_PATH.strip()
            else md_path.with_name(md_path.name + ".preview.html")
        )
        render_markdown(md_path, html_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(f"Markdown: {md_path}")
    print(f"HTML 预览: {html_path}")
    if OPEN_BROWSER:
        webbrowser.open(html_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
