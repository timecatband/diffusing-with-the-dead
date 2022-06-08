import os
import re
os.system("mkdir data")
os.system("curl \"https://archive.org/details/gd1974-05-25.aud.gems.111301.flac16\" > dead.html")
f = open("dead.html")
contents = f.read();
matches = re.findall("\"https[^\"]+\.mp3\"", contents)
matches = [a.replace("\"","") for a in matches]
n=0
print(matches)
for match in matches:
        print(match)
        os.system("curl -LOs " + match)
        n = n+1
os.system("mv *.mp3 data")
