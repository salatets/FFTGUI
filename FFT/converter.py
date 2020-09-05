#convert files with code to c string

lines = []
maxLen = 0

with open('fft.cl','r') as f:
  lines = f.readlines()

for i in range(len(lines)):
  if len(lines[i]) > maxLen:  
    maxLen = len(lines[i])

with open('fft.cl.out','w') as f:
  f.write('"\\n"\\\n') 
  for i in range(len(lines)):
    f.write('"'+lines[i][0:-1] + (" " * (maxLen - len(lines[i]))) + "\\n\" \\\n")

print("done")
