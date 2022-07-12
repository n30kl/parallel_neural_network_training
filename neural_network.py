import asyncio
import time
from random import Random, random
import numpy as np

def function_sigmmoid(x):
   return 1 / (1 + np.exp(-x)) # f(x) = 1 / (1 + e^(-x))
 
def deriv_function_sigmmoid(x):
  fx = function_sigmmoid(x) #f'(x) = f(x) * (1 - f(x))
  return fx * (1 - fx)

def losses(ytrue, ypred):
  return ((ytrue - ypred) ** 2).mean()

class NeuralNetwork:
 
  def __init__(this):
    this.r1 = np.random.normal()
    this.r2 = np.random.normal()
    this.r3 = np.random.normal()
    this.r4 = np.random.normal()
    this.r5 = np.random.normal()
    this.r6 = np.random.normal()

    this.b1 = np.random.normal()
    this.b2 = np.random.normal()
    this.b3 = np.random.normal()

  def feedforward(this, x):
    h1 = function_sigmmoid(this.r1 * x[0] + this.r2 * x[1] + this.b1)
    h2 = function_sigmmoid(this.r3 * x[0] + this.r4 * x[1] + this.b2)
    o1 = function_sigmmoid(this.r5 * h1 + this.r6 * h2 + this.b3)
    return o1

  def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

  @background
  def train(this, data_array, all_y_trues):

    learn_rate = 0.1
    epochs = 1000

    for epoch in range(epochs):
      for x, y_true in zip(data_array, all_y_trues):
        sum_h1 = this.r1 * x[0] + this.r2 * x[1] + this.b1
        h1 = function_sigmmoid(sum_h1)

        sum_h2 = this.r3 * x[0] + this.r4 * x[1] + this.b2
        h2 = function_sigmmoid(sum_h2)

        sum_o1 = this.r5 * h1 + this.r6 * h2 + this.b3
        o1 = function_sigmmoid(sum_o1)
        y_pred = o1

        d_L_d_ypred = -2 * (y_true - y_pred)

        d_ypred_d_r5 = h1 * deriv_function_sigmmoid(sum_o1)
        d_ypred_d_r6 = h2 * deriv_function_sigmmoid(sum_o1)
        d_ypred_d_b3 = deriv_function_sigmmoid(sum_o1)

        d_ypred_d_h1 = this.r5 * deriv_function_sigmmoid(sum_o1)
        d_ypred_d_h2 = this.r6 * deriv_function_sigmmoid(sum_o1)

        d_h1_d_r1 = x[0] * deriv_function_sigmmoid(sum_h1)
        d_h1_d_r2 = x[1] * deriv_function_sigmmoid(sum_h1)
        d_h1_d_b1 = deriv_function_sigmmoid(sum_h1)

        d_h2_d_r3 = x[0] * deriv_function_sigmmoid(sum_h2)
        d_h2_d_r4 = x[1] * deriv_function_sigmmoid(sum_h2)
        d_h2_d_b2 = deriv_function_sigmmoid(sum_h2)

        this.r1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_r1
        this.r2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_r2
        this.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        this.r3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_r3
        this.r4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_r4
        this.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        this.r5 -= learn_rate * d_L_d_ypred * d_ypred_d_r5
        this.r6 -= learn_rate * d_L_d_ypred * d_ypred_d_r6
        this.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(this.feedforward, 1, data_array)
        loss = losses(all_y_trues, y_preds)

while(True):
    data_array = []
    all_y_trues = []
    test_data = int(input("Enter num of test data: "))

    for x in range(int(test_data/2)):
        random1 = (-1)*(np.random.randint(20))
        random2 = (-1)*(np.random.randint(20))
        data_array.append([random1, random2])
        
    for x in range(int(test_data/2)):
        random1 = (np.random.randint(20))
        random2 = (np.random.randint(20))
        data_array.append([random1, random2])

    for x in range(int(test_data/2)):
        all_y_trues.append(1)

    for x in range(int(test_data/2)):
        all_y_trues.append(0)
    
    network = NeuralNetwork()
    start = time.monotonic()
    network.train(data_array, all_y_trues)
    stop = time.monotonic()

    print ('Time spend: ', round((stop-start), 1), 'sec \n')

    Galina = np.array([-7, -3])
    Garik = np.array([20, 2])
    Sasha = np.array([-2, 2])
    Zhenya = np.array([2, -2])
    #print("Galina: %.3f" % network.feedforward(Galina))
    #print("Garik: %.3f" % network.feedforward(Garik))
    #print("Sasha: %.3f" % network.feedforward(Sasha))
    #print("Zhenya: %.3f" % network.feedforward(Zhenya))
