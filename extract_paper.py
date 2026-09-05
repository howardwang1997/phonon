import re, html, sys

with open("vse2_paper.html", "r", encoding="utf-8", errors="ignore") as f:
    t = f.read()

# Remove scripts/styles
t = re.sub(r"<script.*?</script>", " ", t, flags=re.S)
t = re.sub(r"<style.*?</style>", " ", t, flags=re.S)
t = re.sub(r"<noscript.*?</noscript>", " ", t, flags=re.S)
# Keep paragraph breaks
t = re.sub(r"</p>", "\n\n", t, flags=re.I)
t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
t = re.sub(r"</h[1-6]>", "\n\n", t, flags=re.I)
t = re.sub(r"</li>", "\n", t, flags=re.I)
# Strip tags
t = re.sub(r"<[^>]+>", " ", t)
t = html.unescape(t)
# Collapse whitespace per line
lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
lines = [ln for ln in lines if ln]
out = "\n".join(lines)
with open("vse2_paper.txt", "w", encoding="utf-8") as f:
    f.write(out)
print("LEN", len(out))
print(out[:6000])
