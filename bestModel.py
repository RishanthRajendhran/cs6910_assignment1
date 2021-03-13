# Init wandb
import wandb
import seaborn as sn
import pandas as pd
from neuralNetwork import NeuralNetwork
from metrics import accuracy, crossEntropyLoss, MSEloss
from keras.datasets import fashion_mnist
import keras as ke 
import numpy as np
import matplotlib.pyplot as plt

hyperparameter_defaults = dict(
    maxIterations = 10,
    numClasses = 10,
    numLayers = 3,
    numNeurons = 128,
    weightDecay = 0,
    learningRate = 0.0001,
    optimizer = "momentum",
    batchSize = 64,
    weightInitialisation = "random",
    activationFn = "sigmoid",
    gamma = 0.9,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
    validSize = 6000
    )

run = wandb.init(project="cs6910_assignment1", entity="rishanthrajendhran",config=hyperparameter_defaults)
config = wandb.config
wandb.run.name = f"{config.maxIterations}_{config.numLayers}_{config.numNeurons}_{config.weightDecay}_{config.learningRate}_{config.optimizer}_{config.batchSize}_{config.weightInitialisation}_{config.activationFn}"
wandb.run.save(wandb.run.name)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#Normalising data
x_train = (x_train-np.mean(x_train)) / np.std(x_train)
x_test = (x_test-np.mean(x_test)) / np.std(x_test)
randomize = np.arange(len(x_train))
np.random.shuffle(randomize)
x_train = x_train[randomize]
y_train = y_train[randomize]
randomize = np.arange(len(x_test))
np.random.shuffle(randomize)
x_test = x_test[randomize]
y_test = y_test[randomize]

x_valid, y_valid = x_train[-config.validSize:], y_train[-config.validSize:]

def main():

    nn = NeuralNetwork(config.numLayers, config.numClasses, config.weightInitialisation, config.activationFn, config.weightDecay)
    nn.initialiseParams(len(x_train[0])*len(x_train[0]), config.numNeurons)

    sample = np.random.randint(3*len(x_train)/4)
    nn.forwardPropagate(x_train[sample])
    nn.momentumGradDesc(x_train, y_train, config.maxIterations, config.learningRate, config.batchSize, config.gamma)

    predictions = []
    predProbs = []
    test_acc = 0
    test_entropy = 0
    test_mse = 0
    for i in range(len(x_test)):
        nn.forwardPropagate(x_test[i])
        predictions.append(nn.predictedClass)
        predProbs.append(nn.output[nn.predictedClass])


    test_acc = accuracy(y_test,predictions)
    test_entropy = crossEntropyLoss(y_test,predProbs)
    test_mse = MSEloss(y_test,predictions)

    confusion_matrix = np.zeros((config.numClasses, config.numClasses))
    for i in range(len(y_test)):
        confusion_matrix[predictions[i]][y_test[i]] += 1
    
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"], columns = [i for i in "0123456789"])
    plt.figure(figsize = (10,10))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion Matrix")
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    wandb.log({"plot":wandb.Image(plt)})
    plt.show()
    # #Log in wandb
    metrics = {
        'test_acc': test_acc, 
        # 'test_entropy': test_entropy,
        "test_mse": test_mse, 
        # "confusion_matrix": confusion_matrix,
    }
    wandb.log(metrics)
    run.finish()

if __name__ == "__main__":
    main()

