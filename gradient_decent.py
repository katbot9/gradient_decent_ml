import numpy as np
import pandas as pd
import math

df = pd.read_csv(r"C:\Users\dell\Desktop\scores.csv") //make sure to change the directory !!

def gradient_decent(x,y):
    m = b = 0
    n = len(x)
    lrate = 0.000211
    cost = -1
    precost = 0
    i = 0
    while (not math.isclose(cost,precost,rel_tol=1e-20)):
        precost = cost
        y_pred = m*x+b
        cost = 1/n * sum([val**2 for val in (y-y_pred)])
        dm = - 2/n * sum(x * (y-y_pred))
        db = - 2/n * sum( (y-y_pred))
        i+=1
        print("m {}   b {}   cost {}    iteration {}".format(m,b,cost,i))
        m = m - lrate * dm
        b = b - lrate * db

x = np.array(df.math)
y = np.array(df.cs)



gradient_decent(x,y)
