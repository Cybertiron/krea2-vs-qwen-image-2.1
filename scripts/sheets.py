"""Build one comparison sheet per prompt: rows = seeds, columns = variants, labelled with generation time."""
import json, os, glob, sys
from PIL import Image, ImageDraw, ImageFont

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALL = {"krea2": "Krea 2 Turbo fp8 (8 st.)", "krea2int8": "Krea 2 Turbo int8 (8 st.)",
       "qwen21": "Qwen-Image-2.1 (25 st.)", "qwen21fix": "Qwen 2.1 + Fix LoRA (20 st., CFG3)"}
# usage: sheets.py [variant,variant,...] [suffix]
VARS = [(v, ALL[v]) for v in (sys.argv[1] if len(sys.argv) > 1 else "krea2,qwen21,qwen21fix").split(",")]
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else "sheet"
CELL, HEAD = 640, 44
times = json.load(open(os.path.join(D, "timings.json")))
font = ImageFont.truetype("arialbd.ttf", 26)
small = ImageFont.truetype("arial.ttf", 22)

for pr in ["portrait", "landscape", "text", "anime_fantasy", "composition"]:
    seeds = sorted({os.path.basename(f).rsplit("_", 2)[1] for f in glob.glob(os.path.join(D, f"{pr}_[0-9]*_*.png"))})
    if not seeds:
        continue
    sheet = Image.new("RGB", (CELL * len(VARS), HEAD + (CELL + HEAD) * len(seeds)), "white")
    dr = ImageDraw.Draw(sheet)
    for c, (_, title) in enumerate(VARS):
        dr.text((c * CELL + 12, 9), title, fill="black", font=font)
    for r, s in enumerate(seeds):
        y = HEAD + r * (CELL + HEAD)
        for c, (v, _) in enumerate(VARS):
            p = os.path.join(D, f"{pr}_{s}_{v}.png")
            if not os.path.exists(p):
                continue
            sheet.paste(Image.open(p).convert("RGB").resize((CELL, CELL), Image.LANCZOS), (c * CELL, y + HEAD))
            dr.text((c * CELL + 12, y + 10), f"seed {s}  ·  {times.get(f'{pr}_{s}_{v}', 0):.1f} s", fill="#444", font=small)
    out = os.path.join(D, f"{pr}_{SUFFIX}.jpg")
    sheet.save(out, quality=90)
    print(out)

for v, title in VARS:
    ts = [t for k, t in times.items() if k.endswith("_" + v)]
    if ts:
        print(f"{title}: avg {sum(ts) / len(ts):.1f}s over {len(ts)} images")
