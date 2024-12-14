infile = open("diabetesrenewed.csv", "r")
outfile = open("diabetesfeatures.csv", "w")

for line in infile:
    line = line.rstrip()
    linelist = line.split(",")
    linelist.remove(linelist[0])
    linelist.remove(linelist[2])
    linelist.remove(linelist[2])
    
    linestr = ",".join(linelist)
    
    outfile.write(linestr)
    outfile.write("\n")


infile.close()
outfile.close()