from pathlib import Path

# open the script in CancerQML folder


infile_path = Path("./data/diabetes.csv")
outfile_path = Path("./data/diabetesrenewed.csv")


# data cleaner


infile = open(infile_path, "r")
outfile = open(outfile_path, "w")

for line in infile:
    line = line.rstrip()
    linelist = line.split(",")
    if "0" not in linelist[:-1]:
        linelist2 = ",".join(linelist)
        outfile.write(linelist2)
        '''for i in linelist:
            if linelist.index(i) != len(linelist)-1:
                outfile.write(f"{i},")
            else:
                outfile.write(f"{i}")'''
        outfile.write("\n")

infile.close()
outfile.close()