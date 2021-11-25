
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, max_error
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from vmdpy import VMD  
from PyEMD import EMD


# import data 
Stoxx_Europe = pd.read_csv("C:/yourpath/STOXX All Europe 100 EUR Price Historical Data.csv")
Stoxx_Europe = Stoxx_Europe.replace(',','', regex=True)

SP_BSE = pd.read_csv("C:/yourpath/S&P BSE-100 Historical Data.csv")
SP_BSE = SP_BSE.replace(',','', regex=True)

PK_Karachi = pd.read_csv("C:/yourpath/Karachi 100 Historical Data.csv")
PK_Karachi = PK_Karachi.replace(',','', regex=True)


# prepare and standardize data
y = SP_BSE['Price']
y = pd.to_numeric(y.astype(str).str.strip())
y = SP_BSE['Price'].to_numpy() 
y = y.astype(float) 
y = stats.zscore(y)   



# Train_test split
train_split = int(y.size*0.7)
y_train = y[:train_split]
y_test = y[train_split:]




# EMD  
u = y
emd = EMD()
u = emd(u)

u_train = u[:, :train_split]
u_test = u[:, train_split:]




# Visualize decomposed modes for train data
plt.figure()
plt.subplot(2,1,1)
plt.plot(y_train)
plt.title('Original train signal')
plt.xlabel('time (s)')
plt.subplot(2,1,2)
plt.plot(u_train.T)
plt.title('Decomposed train modes')
plt.xlabel('time (s)')
plt.legend(['Mode %d'%m_i for m_i in range(u_train.shape[0])])
plt.tight_layout()
plt.show()

# Visualize decomposed modes for test data
plt.figure()
plt.subplot(2,1,1)
plt.plot(y_test)
plt.title('Original test signal')
plt.xlabel('time (s)')
plt.subplot(2,1,2)
plt.plot(u_test.T)
plt.title('Decomposed test modes')
plt.xlabel('time (s)')
plt.legend(['Mode %d'%m_i for m_i in range(u_test.shape[0])])
plt.tight_layout()
plt.show()


# SVR
reg = SVR()

params = {'C': stats.uniform(loc=0, scale=2),
          'kernel': ['linear']}

reg_cv = RandomizedSearchCV(reg, param_distributions=params, n_iter=50, 
                            scoring="neg_mean_absolute_error")
reg_cv.fit(u_train.T, y_train[0:])
y_pred = reg_cv.predict(u_test.T)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
maxerror = max_error(y_test, y_pred)


print(f"Averaged MAE: {mae:.5f}")
print(f"Residual error: {R2:.5f}")
print(f"Maximum Residual error: {maxerror:.5f}")

