from pathlib import Path

import pandas as pd,numpy as np, matplotlib.pyplot as plt
what_is_this,c2,spectrum = pd.read_csv("data//DOPPWA02.CSV").to_numpy().T
#plt.plot(what_is_this, label ="c1")
#plt.plot(c2,label = "c2")
plt.plot(spectrum, label ="c3")
plt.legend()
plt.show()