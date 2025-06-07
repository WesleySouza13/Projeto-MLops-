from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

x,y = load_iris(return_X_y=True)

# split 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)
print(f'shape variaveis de treino:{x_train.shape}e { y_train.shape}')

print(f'shape variaveis de teste:{x_test.shape}e { y_test.shape}')

# treino e avaliaçao 
model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)

# treino 
acc_train = accuracy_score(y_pred= y_train_pred, y_true=y_train)
print(f'acuracia treino:{acc_train}')

# teste 
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc_test = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f'acuracia teste: {acc_test}')

# salvando modelo 
with open('modelo_reg_logistica', 'wb') as f:
    joblib.dump('modelo_reg_logistica', f)