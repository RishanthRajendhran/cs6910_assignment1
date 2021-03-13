# Init wandb
import wandb
hyperparameter_defaults = dict(
    maxIterations = 5,
    numClasses = 10,
    numLayers = 3,
    numNeurons = 32,
    weightDecay = 0,
    learningRate = 1e-3,
    optimizer = "sgd",
    batchSize = 16,
    weightInitialisation = "xavier",
    activationFn = "tanh",
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

from neuralNetwork import NeuralNetwork
from metrics import accuracy, crossEntropyLoss, MSEloss

from keras.datasets import fashion_mnist
import keras as ke 
import numpy as np
import matplotlib.pyplot as plt

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
# x_test, y_test = x_test[:40], y_test[:40]
# x_valid, y_valid = x_valid[:40], y_valid[:40]


def visualiseTrainingExamples():
    x_unique = []
    y_unique = []
    for i in range(x_train.shape[0]):
        if y_train[i] not in y_unique:
            y_unique.append(y_train[i])
            x_unique.append(x_train[i])
        if len(y_unique) == 10:
            break

    x_unique, y_unique = (list(t) for t in zip(*sorted(zip(x_unique, y_unique), key = lambda rec : rec[1])))

    for i in range(len(x_unique)):
        plt.imshow(x_unique[i])
        plt.title(f"Class {y_unique[i]}")
        wandb.log({"plot:":plt},step=i)

def main():

    visualiseTrainingExamples()

    nn = NeuralNetwork(config.numLayers, config.numClasses, config.weightInitialisation, config.activationFn, config.weightDecay)
    nn.initialiseParams(len(x_train[0])*len(x_train[0]), config.numNeurons)

    sample = np.random.randint(3*len(x_train)/4)
    nn.forwardPropagate(x_train[sample])
    if config.optimizer == "sgd":
        nn.stochasticGradDesc(x_train, y_train, config.maxIterations, config.learningRate, config.batchSize)
    elif config.optimizer == "momentum":
        nn.momentumGradDesc(x_train, y_train, config.maxIterations, config.learningRate, config.batchSize, config.gamma)
    elif config.optimizer == "nesterov":
        nn.nesterovAcceleratedGradDesc(x_train, y_train, config.maxIterations, config.learningRate, config.batchSize, config.gamma)
    elif config.optimizer == "rmsprop":
        nn.rmsprop(x_train, y_train, config.maxIterations, config.learningRate, config.batchSize, config.eps, config.beta1)
    elif config.optimizer == "adam":
        nn.adam(x_train, y_train, config.maxIterations, config.learningRate, config.batchSize, config.eps, config.beta1, config.beta2)
    elif config.optimizer == "nadam":
        nn.nadam(x_train, y_train, config.maxIterations, config.learningRate, config.batchSize, config.eps, config.beta1, config.beta2)
    else:
        print("No such optimizer available.")

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

    predictions = []
    predProbs = []
    valid_acc = 0
    valid_entropy = 0
    valid_mse = 0
    for i in range(len(x_valid)):
        nn.forwardPropagate(x_valid[i])
        predictions.append(nn.predictedClass)
        predProbs.append(nn.output[nn.predictedClass])

    valid_acc = accuracy(y_valid,predictions)
    valid_entropy = crossEntropyLoss(y_valid,predProbs)
    valid_mse = MSEloss(y_valid,predictions)

    print(f"Test Set:\nAccuracy = {test_acc}\nLoss = {test_entropy}\nMSE = {test_mse}")
    print(f"Train Set:\nAccuracy = {valid_acc}\nLoss = {valid_entropy}\nMSE = {valid_mse}")

    # #Log in wandb
    metrics = {
        'test_acc': test_acc, 
        # 'test_entropy': test_entropy,
        "test_mse": test_mse, 
        'valid_acc': valid_acc,
        # 'valid_entropy': valid_entropy, 
        "valid_mse": valid_mse
    }
    wandb.log(metrics)
    run.finish()

if __name__ == "__main__":
    main()

