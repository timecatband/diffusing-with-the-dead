import os
import re
#os.system("mkdir data")
#os.system("curl \"https://archive.org/details/gd1974-05-25.aud.gems.111301.flac16\" > dead.html")
#os.system("curl \"https://archive.org/details/gd1982-05-23.156110.senn421.vita.miller.clugston.flac1648/gd82-05-23+s1t01+Shakedown.flac\" > dead1.html")
#os.system("curl \"https://archive.org/details/gd1987-09-07.156739.nak300.carpenter.flac1644\" > dead2.html")
#os.system("curl \"https://archive.org/details/gd1973-05-26.147312.aud.taback.flac16/s02t09+-+Bertha.flac\" > dead3.html")
#os.system("curl \"https://archive.org/details/gd1967-07-23.aud.sorochty.125462.flac16\" > dead4.html")
os.system("curl \"https://archive.org/details/gd1969-06-11.145953.aud.flac1644\" > dead5.html")
def get_mp3s(html):
    f = open(html)
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
get_mp3s("dead5.html")
