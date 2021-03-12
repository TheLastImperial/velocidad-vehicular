import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt

import TLI.preprocess as pp

model = tf.keras.models.load_model('resources/no_angle.h5')
X_train, X_test, Y_train, Y_test = pp.load_simple_no_angle("resources/result3.result")

Y_predict = model.predict(X_test)
Y_predict *= 92
Y_test *= 92


Y_predict = Y_predict.flatten()
X = list(range(len(Y_test)))

print(Y_test)
print("----------------------")
print(Y_predict)
print("----------------------")
print(X)


plt.figure("Figura MAE")
plt.xlabel('Cantidad')
plt.ylabel('Kilometros')

plt.plot(X, Y_test,
       label='RIGHT', color="blue",
       linestyle="", marker="o")
plt.plot(X, Y_predict,
       label = 'WRONG', color="red",
       linestyle="", marker="o")

# plt.subplot(X, Y_test,
#        label='RIGHT', color="blue")
# plt.plot(0,1)
# plt.subplot(X, Y_predict, 1,
#        label = 'WRONG', color="red")
# plt.plot(1,1)

plt.legend()
plt.show()
