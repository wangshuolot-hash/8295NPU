import re, sys

def extract(path, keyword_pat):
    with open(path, 'rb') as f:
        data = f.read()
    # printable ASCII runs >= 6 chars
    runs = re.findall(rb'[\x20-\x7e]{6,}', data)
    hits = set()
    pat = re.compile(keyword_pat, re.IGNORECASE)
    for r in runs:
        s = r.decode('ascii', 'ignore')
        if pat.search(s) and len(s) < 120:
            hits.add(s)
    return hits

sdk = r"C:\产品资料\AI\v2.50.0.260828\qairt\2.50.0.260828\lib"
files = {
    'v68': sdk + r"\hexagon-v68\unsigned\libQnnHtpV68Skel.so",
    'v69': sdk + r"\hexagon-v69\unsigned\libQnnHtpV69Skel.so",
}

matmul68 = extract(files['v68'], r'matmul')
matmul69 = extract(files['v69'], r'matmul')
print("=== v68 matmul ops (%d) ===" % len(matmul68))
for s in sorted(matmul68):
    print(" ", s)
print("=== v69 matmul ops (%d) ===" % len(matmul69))
for s in sorted(matmul69):
    print(" ", s)
print("=== only in v69 (missing from v68) ===")
for s in sorted(matmul69 - matmul68):
    print(" ", s)
