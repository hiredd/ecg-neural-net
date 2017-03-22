import random
f = open("verify.csv", "w")
for _ in range(900):
    x = random.random()*100-50
    y = random.random()*100-50
    z = random.random()*100-50
    val = 1 if x+y+z>0 else -1
    f.write(str(x)+","+str(y)+","+str(z)+","+str(val)+"\n")
f.close()