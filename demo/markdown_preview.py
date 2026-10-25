"""Offline, script-free OCR preview; source Markdown is never modified."""

import re
import html
from html.parser import HTMLParser


class _PreviewHTML(HTMLParser):
    """Keep document formatting, but no executable or network-active markup."""

    tags = set(('p div span h1 h2 h3 h4 h5 h6 ul ol li dl dt dd blockquote '
                'pre code em strong b i s sup sub del ins u hr br small '
                'table thead tbody tfoot tr th td caption colgroup col details summary').split())
    blocked = {'script', 'style', 'iframe', 'object', 'svg', 'math', 'template'}

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.output = []
        self.suppressed = []

    def handle_starttag(self, tag, attrs):
        if self.suppressed or tag in self.blocked:
            self.suppressed.append(tag)
            return
        if tag not in self.tags:
            return
        safe = []
        for name, value in attrs:
            if value is None:
                continue
            if name in {'colspan', 'rowspan', 'span', 'start'} and re.fullmatch(r'\d{1,4}', value):
                safe.append((name, value))
            elif name == 'align' and value.lower() in {'left', 'center', 'right', 'justify'}:
                safe.append((name, value.lower()))
            elif name == 'style':
                declarations = []
                for declaration in value.split(';'):
                    prop, separator, setting = declaration.partition(':')
                    allowed = {
                        'text-align': {'left', 'center', 'right', 'justify'},
                        'vertical-align': {'top', 'middle', 'bottom', 'baseline'},
                        'font-weight': {'normal', 'bold', '400', '600', '700'},
                        'font-style': {'normal', 'italic'},
                        'white-space': {'normal', 'pre-wrap', 'pre-line'},
                    }
                    prop, setting = prop.strip().lower(), setting.strip().lower()
                    if separator and setting in allowed.get(prop, set()):
                        declarations.append(prop + ':' + setting)
                if declarations:
                    safe.append(('style', ';'.join(declarations)))
        attributes = ''.join(' ' + key + '="' + html.escape(value, quote=True) + '"'
                             for key, value in safe)
        self.output.append('<' + tag + attributes + '>')

    def handle_startendtag(self, tag, attrs):
        if not self.suppressed and tag not in self.blocked:
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if self.suppressed:
            if tag in self.suppressed:
                self.suppressed = self.suppressed[:self.suppressed.index(tag)]
        elif tag in self.tags:
            self.output.append('</' + tag + '>')

    def handle_data(self, data):
        if not self.suppressed:
            self.output.append(html.escape(data, quote=False))

    def handle_entityref(self, name):
        if not self.suppressed:
            self.output.append('&' + name + ';')

    def handle_charref(self, name):
        if not self.suppressed:
            self.output.append('&#' + name + ';')


STYLE = """
*{box-sizing:border-box}body{margin:0;padding:24px;color:#202b31;background:#fff;
font:15px/1.8 'Segoe UI','Microsoft YaHei',sans-serif;overflow-wrap:anywhere}
h1,h2,h3,h4,h5,h6{line-height:1.4;margin:1.3em 0 .6em}h1{font-size:26px}
h2{font-size:22px}h3{font-size:18px}p{margin:.8em 0;white-space:pre-wrap}
table{border-collapse:collapse;max-width:100%;margin:1em 0}th,td{border:1px solid #b9c5cb;
padding:7px 12px;vertical-align:top;white-space:pre-wrap}th{background:#f0f4f5}
pre{background:#f4f6f7;padding:14px;overflow:auto;white-space:pre-wrap}
code{font-family:Consolas,monospace}blockquote{margin-left:0;padding-left:16px;
border-left:3px solid #91a6b1}a{color:#245b83}hr{border:0;border-top:1px solid #ddd}
@media(max-width:500px){body{padding:14px;font-size:14px}}
"""


def render_markdown(source):
    import markdown

    # Preserve TeX delimiters/backslashes that Markdown would otherwise consume.
    formulas = []

    def hold_formula(match):
        formulas.append(html.escape(match.group(0)))
        return f'YADOCRMATHPLACEHOLDER{len(formulas) - 1}END'

    protected = re.sub(r'\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$', hold_formula,
                       source, flags=re.DOTALL)
    body = markdown.markdown(
        protected, extensions=['extra', 'sane_lists', 'nl2br'], output_format='html5')
    for index, formula in enumerate(formulas):
        body = body.replace(f'YADOCRMATHPLACEHOLDER{index}END', formula)
    sanitized = _PreviewHTML()
    sanitized.feed(body)
    sanitized.close()
    body = ''.join(sanitized.output)
    # The iframe is additionally sandboxed by the parent. No model-produced HTML
    # may execute scripts, load remote assets, submit forms, or navigate the app.
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="'
            "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
            '"><style>' + STYLE + '</style></head><body>' + body + '</body></html>')
