import numpy as np
from sklearn import preprocessing

import TLI.preprocess as pp
import TLI.posprocess as posp
import TLI.neural_network as nn

result = np.array(pp.csv_to_list("resources/result3.result"))
r_t = result.T
X = r_t[:-1].T

# X = X[4:].T
Y = r_t[-1]
X = preprocessing.normalize(X, norm="max")
Y = preprocessing.normalize([Y], norm="max")

X_train, X_test, Y_train, Y_test = pp.split_dataset(X, Y[0])

EPOCHS = 100
model = nn.model_simple_complete()
history = model.fit(
  X_train, Y_train,
  epochs=EPOCHS, validation_split = 0.2,
  verbose=1)
model.save("resources/angle.h5")

Y_predict = model.predict(X_test)

posp.get_metrics(Y_test, Y_predict)

posp.plot_history(history)
