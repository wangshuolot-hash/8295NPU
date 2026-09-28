import re
base = r'C:\Projects\llm-npu\llama-qnn\llama.cpp-dev-refactoring\ggml\src\ggml-qnn'
import os, sys
for root, dirs, files in os.walk(base):
    for fn in files:
        if not fn.endswith(('.cpp', '.hpp')):
            continue
        p = os.path.join(root, fn)
        src = open(p, encoding='utf-8', errors='ignore').read()
        for i, l in enumerate(src.splitlines()):
            if ('HTP' in l or 'kQnnDeviceName' in l or 'kQnnBackendName' in l
                or 'qnn_device_name' in l or ('_name ' in l and 'QNN' in l)
                or re.search(r'"[Qq][Nn][Nn]', l)):
                print(os.path.relpath(p, base) + ':' + str(i+1) + ': ' + l.strip())
