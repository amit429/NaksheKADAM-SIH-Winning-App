import os
replacement = "mentyour"
for dname, dirs, files in os.walk("./lib"):
    for fname in files:
        fpath = os.path.join(dname, fname)
        print(fpath)
        with open(fpath, "r") as f:
            s = f.read()
        s = s.replace("mentYOUR", replacement)
        with open(fpath, "w") as f:
            f.write(s)