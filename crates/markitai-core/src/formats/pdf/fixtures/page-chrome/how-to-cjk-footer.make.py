"""Invented 3-page how-to PDF printed by Chromium: screenshots, bold step lines, running footer + page numbers."""
import base64, io, sys
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

def shot(w, h, title, seed):
    im = Image.new("RGB", (w, h), (250, 244, 240))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 40], fill=(60, 60, 70))
    d.text((12, 12), title, fill=(255, 255, 255))
    for i in range(6):
        y = 70 + i * (h - 100) // 6
        d.rectangle([20 + seed * 3, y, w - 30, y + 30], outline=(200, 120, 90), width=3)
        d.text((30 + seed * 3, y + 8), f"Option {i + 1}", fill=(40, 40, 40))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

steps = [
    ("1", "打开<b>设置</b>面板，选择<b>网络</b>。"),
    ("2", "列表加载后，在 <b>Proxy</b> 分组选择 1 个你要使用的服务器"),
    ("3", "最后，在<b>控制台</b>右下角点击<b>启动按钮</b>开启服务，就完成了"),
]
html = f"""<html><head><meta charset=utf-8><style>
body {{ font-family: 'DejaVu Sans', sans-serif; margin: 0; }}
h1 {{ font-size: 28px; }} p {{ font-size: 15px; }} img {{ width: 100%; }}
p.s {{ font-family: 'Noto Sans CJK SC', 'Noto Serif CJK SC'; font-size: 16px; }}
.pb {{ page-break-before: always; }}
</style></head><body>
<h1>Desk Lamp Proxy Guide</h1>
<p>This guide walks through the three steps to set up the desk lamp proxy client on a laptop.</p>
<p>Step one: download and install the client from the vendor page, then start it once.</p>
<p>Step two: import the settings file the support desk sent you by email.</p>
<div class=pb></div>
<img src="{shot(1536, 766, 'Client', 1)}">
<p class=s>{steps[1][0]}、{steps[1][1]}</p>
<img src="{shot(1536, 992, 'Dashboard', 2)}">
<p class=s>{steps[2][0]}、{steps[2][1]}</p>
<div class=pb></div>
<img src="{shot(1536, 986, 'Status', 3)}">
</body></html>"""
footer = """<div style="font-size:9px;width:100%;padding:0 36px;display:flex;justify-content:space-between;font-family:'DejaVu Sans'">
<span>Desk Lamp Proxy Guide</span><span class="pageNumber"></span></div>"""
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.set_content(html)
    pg.pdf(path=sys.argv[1], format="A4", display_header_footer=True, header_template="<span></span>",
           footer_template=footer, margin={"top": "72px", "bottom": "60px", "left": "96px", "right": "96px"})
    b.close()
