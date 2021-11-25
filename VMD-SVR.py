
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, max_error
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from vmdpy import VMD  

# import data 
Stoxx_Europe = pd.read_csv("C:/yourpath/STOXX All Europe 100 EUR Price Historical Data.csv")
Stoxx_Europe = Stoxx_Europe.replace(',','', regex=True)

SP_BSE = pd.read_csv("C:/yourpath/S&P BSE-100 Historical Data.csv")
SP_BSE = SP_BSE.replace(',','', regex=True)

PK_Karachi = pd.read_csv("C:/yourpath/Karachi 100 Historical Data.csv")
PK_Karachi = PK_Karachi.replace(',','', regex=True)


# prepare and standardize data
y = PK_Karachi['Price']
y = pd.to_numeric(y.astype(str).str.strip())
y = PK_Karachi['Price'].to_numpy() 
y = y.astype(float) 
y = stats.zscore(y)   

# Train_test split
train_split = int(y.size*0.7)
y_train = y[:train_split]
y_test = y[train_split:]


# VMD  
# parameters for VMD  
alpha = 2000       # moderate bandwidth constraint  
tau = 0.           # noise-tolerance (no strict fidelity enforcement)  
K = 4              # modes IMFs set in advance (change this parameter at each run, I have tested from 3 to 20 IMFs, 
				   # and reported from 3 to 8 IMFs as these produce the most significant results)
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  	   # tolerance in exponential terms

# Decomposition on train data
# Time Domain 0 to T  
T = y_train.size
fs = 1/T  
t = np.arange(1,T+1)/T  
freqs = 2*np.pi*(t-0.5-fs)/(fs)  

# Run actual VMD code  
u_train, u_hat_train, omega_train = VMD(y_train, alpha, tau, K, DC, init, tol) 

# Decomposition on test data
# Time Domain 0 to T  
T = y_test.size
fs = 1/T  
t = np.arange(1,T+1)/T  
freqs = 2*np.pi*(t-0.5-fs)/(fs)  

# Run actual VMD code  
u_test, u_hat_test, omega_test = VMD(y_test, alpha, tau, K, DC, init, tol) 


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
reg_cv.fit(u_train.T, y_train[1:])
y_pred = reg_cv.predict(u_test.T)

# Evaluation
mae = mean_absolute_error(y_test[1:], y_pred)
R2 = r2_score(y_test[1:], y_pred)
maxerror = max_error(y_test[1:], y_pred)


print(f"Averaged MAE: {mae:.5f}")
print(f"Residual error: {R2:.5f}")
print(f"Maximum Residual error: {maxerror:.5f}")

