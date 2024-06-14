import numpy as np
from neuron import Neuron, sigmoid, sigmoid_prime, mse_loss
    
class OurNeuralNetwork():
    def __init__(self) -> None:
        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)

        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, data, y_trues):

        lr = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                dL_dypred = -2 * (y_true - y_pred)

                dypred_dw5 = h1 * sigmoid_prime(sum_o1) # neuron o1
                dypred_dw6 = h2 * sigmoid_prime(sum_o1)
                dypred_db3 = sigmoid_prime(sum_o1)
                dypred_dh1 = self.w5 + sigmoid_prime(sum_o1)
                dypred_dh2 = self.w6 + sigmoid_prime(sum_o1)

                dh1_dw1 = x[0] * sigmoid_prime(sum_h1) # neuron h1
                dh1_dw2 = x[1] * sigmoid_prime(sum_h1)
                dh1_db1 = sigmoid_prime(sum_h1)

                dh2_dw3 = x[0] * sigmoid_prime(sum_h2) # neuron h2
                dh2_dw4 = x[1] * sigmoid_prime(sum_h2)
                dh2_db2 = sigmoid_prime(sum_h2)

                
                self.w1 -= lr * dL_dypred * dypred_dh1 * dh1_dw1
                self.w2 -= lr * dL_dypred * dypred_dh1 * dh1_dw2
                self.b1 -= lr * dL_dypred * dypred_dh1 * dh1_db1

                self.w3 -= lr * dL_dypred * dypred_dh2 * dh2_dw3
                self.w4 -= lr * dL_dypred * dypred_dh2 * dh2_dw4
                self.b2 -= lr * dL_dypred * dypred_dh2 * dh2_db2

                self.w5 -= lr * dL_dypred * dypred_dw5
                self.w6 -= lr * dL_dypred * dypred_dw6
                self.b3 -= lr * dL_dypred * dypred_db3

                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(y_trues, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss))

data = np.array([
  [60-70, 165-155],  # Alice
  [72.5-70, 183-155],   # Bob
  [69-70, 178-155],   # Charlie
  [54.5-70, 152.5-155], # Diana
  [84.5-70, 150-155], 
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
  1
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

sample_wt = 81
sample_ht = 150

sample = np.array([sample_wt-70, sample_ht-155]) 

print(network.feedforward(sample))
if network.feedforward(sample) > 0.5:
    print("female")
else:
    print("male")


                




