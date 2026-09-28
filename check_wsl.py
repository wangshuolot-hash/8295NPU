import subprocess, sys
r = subprocess.run(['wsl', '-l', '-v'], capture_output=True)
data = r.stdout + r.stderr
text = data.decode('utf-16-le', errors='replace')
print('--- wsl -l -v ---')
print(text)
r2 = subprocess.run(['wsl', 'echo', 'WSL_OK'], capture_output=True)
t2 = (r2.stdout + r2.stderr)
print('--- wsl echo test ---')
print(t2.decode('utf-16-le', errors='replace') if b'\x00' in t2[:40] else t2.decode('utf-8', errors='replace'))
print('exit:', r2.returncode)
