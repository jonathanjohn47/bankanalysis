import pandas as pd
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
file = pd.read_csv("bank-additional-full.csv", delimiter = ';')
count = 0
for column in file.columns:
	print(column, end=' ')
	count = count + 1
print(count)

X = file.iloc[:, :-1].values
y = file.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
for i in range(0,19,1):				#'k' is the last row in the X array created using iloc function
	X[:,i] = le.fit_transform(X[:,i])
'''
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [19])
X = onehotencoder.fit_transform(X).toarray()'''
X = DataFrame(X)
print(X)

Y = []
for obj in y:
	if(obj == 'yes'):
		Y.append(1)
	elif(obj == 'no'):
		Y.append(0)
	else:
		pass

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

'''
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X = sc_X.fit_transform(X)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)'''

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Random Forest")


from sklearn.svm import SVC
svm = SVC(kernel = "poly", random_state = 0)
y_pred = svm.fit(X_train, y_train)

cm = confusion_matrix(y_test, svm.predict(X_test))
print(cm)
print("SVC")
