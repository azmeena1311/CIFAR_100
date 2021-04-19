import pickle
import numpy as np
import matplotlib.pyplot as plt

filex = open(r'C:\Users\Nidhi\loss_validation.p', 'rb')

data = pickle.load(filex)
filex.close()

data = np.array(data)

loss = data[:, 0]
accuracy = data[:, 1] * 100


plt.figure()
plt.title("Rate of Loss Value w.r.t Epochs")
plt.xticks()
plt.yticks()
plt.xlim(0, len(loss)+20)
plt.ylim(0, max(loss)+1)
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.plot(loss)
plt.show()

plt.figure()
plt.title("Rate of Accuracy w.r.t Epochs")
plt.xticks()
plt.yticks()
plt.xlim(0, len(accuracy)+20)
plt.ylim(0, 100)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (in %)')
plt.plot(accuracy)
plt.show()