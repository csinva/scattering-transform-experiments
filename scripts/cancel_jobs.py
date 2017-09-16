import subprocess
s = 31286
e = 31292
for j in range(s, e+1):
    subprocess.call("scancel " + str(j), shell=True)
