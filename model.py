import wooldridge as wd
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

wage = wd.data('wage1')[['wage', 'educ', 'exper', 'female']]

y = wage['wage']
x = wage.drop(columns='wage')

model = LinearRegression()
model.fit(x, y)

with open("model.pickle", "wb") as doc:
    pickle.dump(model, doc)

print(model.predict(np.array([[10, 10, 1]])))
