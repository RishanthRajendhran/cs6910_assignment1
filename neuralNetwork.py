import numpy as np
import keras as ke

class NeuralNetwork:
  def __init__(self, numLayers=3, numClasses=10, initialisation="random", activationFn="sigmoid", weightDecay = 0):
    if numLayers <= 1:
        print("NeuralNetwork with 1 (or less) layers not supported")
        exit(0)
    self.numLayers = numLayers
    self.numClasses = numClasses
    self.numNeurons = []
    self.scaleDownFactor = 100000
    self.weights = []
    self.gradW = []
    self.gradB = []
    self.biases = []
    self.initialisation = initialisation
    self.activationFn = activationFn
    self.weightDecay = weightDecay
    self.input = []
    self.preactivations = []
    self.activations = []
    self.output = []
    self.predictedClass = 0

  def initialiseParams(self,size,numNeus=-1):
    if numNeus == -1:
      numNeus = size
    self.numNeurons = []
    self.numNeurons.append(size)
    for i in range(self.numLayers-2):
      self.numNeurons.append(numNeus)
    self.numNeurons.append(self.numClasses)
    self.weights = []
    self.biases = []
    if self.initialisation == "random":
      for i in range(self.numLayers):
        curW = []
        prev = i - (i!=0)
        for j in range(self.numNeurons[i]):
          curW.append((np.random.rand(self.numNeurons[prev],)/(self.scaleDownFactor*(5**i))).tolist())
        curB = (np.random.rand(self.numNeurons[prev],)/(self.scaleDownFactor*(5**i))).tolist()
        self.weights.append(curW)
        self.biases.append(curB)
    elif self.initialisation == "xavier":
      for i in range(self.numLayers):
        curW = []
        prev = i - (i!=0)
        for j in range(self.numNeurons[i]):
          curW.append(((ke.initializers.GlorotNormal(seed=None)(shape=(self.numNeurons[prev],)).numpy())/(self.scaleDownFactor*(10**i))).tolist())
        curB = ((ke.initializers.GlorotNormal(seed=None)(shape=(self.numNeurons[prev],)).numpy())/(self.scaleDownFactor*(10**i))).tolist()
        self.weights.append(curW)
        self.biases.append(curB)
    self.input = []
    self.output = []
    self.activations = []
    self.preactivations = []
    self.output = []
    self.gradW, self.gradB = self.zeroMatrices()
    # self.normaliseWeightsAndBiases()

  def normalisePreacts(self, layer):
    mean = np.mean(self.preactivations[layer])
    std = np.std(self.preactivations[layer])
    self.preactivations[layer] = (self.preactivations[layer] - mean)/std

  def normaliseActs(self, layer):
    mean = np.mean(self.activations[layer])
    std = np.std(self.activations[layer])
    self.activations[layer] = (self.activations[layer] - mean)/std

  def normaliseWeightsAndBiases(self):
    for i in range(self.numLayers):
      meanB = np.mean(self.biases[i])
      stdB = np.std(self.biases[i])
      for j in range(self.numNeurons[i]):
        meanW = np.mean(self.weights[i])
        stdW = np.std(self.weights[i])
        self.biases[i][j] = (self.biases[i][j]-meanB)/stdB
        for k in range(len(self.weights[i][j])):
          self.weights[i][j][k] = (self.weights[i][j][k]-meanW)/stdW

  def processInput(self,input):
    return np.ndarray.flatten(input).tolist()

  def preactivate(self, weight, bias, input):
    return (bias + np.dot(weight,input))

  def activate(self, preactivation):
    if self.activationFn == "sigmoid":
      return 1./(1.+np.exp(-preactivation,dtype=np.float128))
    elif self.activationFn == "tanh":
      return np.tanh(preactivation,dtype=np.float128)
    elif self.activationFn == "ReLU":
      return np.max([0,preactivation])
    return 0

  def finalLayer(self):
    outputs = [self.preactivations[self.numLayers-1][i] for i in range(self.numClasses)]
    sum = 0
    for i in range(len(outputs)):
      outputs[i] = np.exp((outputs[i]),dtype=np.float128)
      sum = sum + outputs[i]
    outputs = np.array(outputs)
    outputs = (np.ndarray.flatten(outputs)/sum).tolist()
    self.output = outputs
    self.predictedClass = np.argmax(self.output)

  def forwardPropagate(self,input):
    input = self.processInput(input)
    if len(self.weights) == 0:
      print("forwardPropagate() called before initialisation")
      self.initialiseParams(len(input))
    self.input = input
    self.preactivations = []
    self.activations = []
    for i in range(self.numLayers):
      pres = []
      activs = []
      for j in range(self.numNeurons[i]):
        preactivation = self.preactivate(self.weights[i][j],self.biases[i][j],input)
        pres.append(preactivation)
      self.preactivations.append(pres)
      self.normalisePreacts(i) 
      for j in range(self.numNeurons[i]):
        activs.append(self.activate(self.preactivations[i][j]))
      self.activations.append(activs)
      self.normaliseActs(i)
      input = activs
    self.finalLayer()

  def oneHot(self, y):
    out = []
    for i in range(self.numClasses):
      if i == y:
        out.append(1)
      else:
        out.append(0)
    return out

  def g_dash(self, preactivation):
    if self.activationFn == "sigmoid":
      activation = self.activate(preactivation)
      return activation*(1-activation)
    elif self.activationFn == "tanh":
      activation = self.activate(preactivation)
      return (1 - (activation**2))
    elif self.activationFn == "ReLU":
      activation = self.activate(preactivation)
      return (activation > 0)
    return 0

  def updateGradW(self, gradW):
    if(len(self.gradW)):
      for i in range(self.numLayers):
        for j in range(self.numNeurons[i]):
            for k in range(len(gradW[i][j])):
                self.gradW[i][j][k] = self.gradW[i][j][k] + gradW[i][j][k]
    else:
      self.gradW = gradW
  
  def updateGradB(self, gradB):
    if(len(self.gradB)):
      for i in range(self.numLayers):
        for j in range(self.numNeurons[i]):
          self.gradB[i][j] = self.gradB[i][j] + gradB[i][j] 
    else:
      self.gradB = gradB

  def backwardPropagate(self, input, y):
    if self.input != self.processInput(input):
      print("backwardPropagate() called before forwardPropagate")
      self.forwardPropagate(input)

    grad_a = [[] for i in range(self.numLayers)]
    grad_W = [[] for i in range(self.numLayers)]
    grad_b = [[] for i in range(self.numLayers)]
    grad_h = [[] for i in range(self.numLayers)]

    grad_a[self.numLayers-1] = -1 * (np.subtract(self.oneHot(y), self.output))
    grad_a[self.numLayers-1] = grad_a[self.numLayers-1].reshape((grad_a[self.numLayers-1].shape[0],1))
    for k in range(self.numLayers-1,-1,-1):
      if k == 0:
        grad_W[k] = np.dot(grad_a[k],np.transpose(np.array(input).reshape(np.array(input).shape[0]*np.array(input).shape[1],1)))
        grad_b[k] = grad_a[k]
      else:
        grad_W[k] = np.dot(grad_a[k],np.transpose(np.array(self.activations[k-1]).reshape((len(self.activations[k-1]),1))))
        grad_b[k] = grad_a[k]
        grad_h[k-1] = np.dot(np.transpose(np.array(self.weights[k])),grad_a[k])
        if k == 1:
          g_dash_list = [self.g_dash(preactivation) for preactivation in self.input]
          grad_a[k-1] = np.multiply(grad_h[k-1],np.array(g_dash_list).reshape(len(g_dash_list),1))
        else:
          g_dash_list = [self.g_dash(preactivation) for preactivation in self.preactivations[k-1]]
          grad_a[k-1] = np.multiply(grad_h[k-1],np.array(g_dash_list).reshape((len(g_dash_list),1)))
    self.updateGradW(grad_W)
    self.updateGradB(grad_b)
    del grad_W
    del grad_b
    del grad_a
    del grad_h

  def updateWeights(self,eta,v=[]):
    if len(v) == 0:
      for i in range(self.numLayers):
        for j in range(self.numNeurons[i]):
            for k in range(len(self.gradW[i][j])):
                self.weights[i][j][k] = self.weights[i][j][k] - eta*self.gradW[i][j][k]
    else:
      for i in range(self.numLayers):
        for j in range(self.numNeurons[i]):
            for k in range(len(self.gradW[i][j])):
                self.weights[i][j][k] = self.weights[i][j][k] - (v[i][j][k] + eta*self.gradW[i][j][k])

  def updateBiases(self,eta,v=[]):
    if len(v) == 0:
      for i in range(self.numLayers):
        for j in range(self.numNeurons[i]):
          self.biases[i][j] = self.biases[i][j] - eta*self.gradB[i][j]
    else:
      for i in range(self.numLayers):
        for j in range(self.numNeurons[i]):
          self.biases[i][j] = self.biases[i][j] - (v[i][j] + eta*self.gradB[i][j])

  def updateVelocityWB(self, prev_v_w, prev_v_b, gamma, eta):
    v_w, v_b = self.zeroMatrices()
    for i in range(self.numLayers):
        for j in range(self.numNeurons[i]):
            v_b[i][j] = gamma*prev_v_b[i][j]
            for k in range(len(prev_v_w[i][j])):
                v_w[i][j][k] = gamma*prev_v_w[i][j][k]
                if len(self.gradW) != 0:
                    v_w[i][j][k] = v_w[i][j][k] + eta*self.gradW[i][j][k]
            if len(self.gradB) != 0:
                v_b[i][j] = v_b[i][j] + eta*self.gradB[i][j]
    return (v_w, v_b)

  def zeroMatrices(self):
    outW, outB = [], []
    for i in range(self.numLayers):
        curW = []
        prev = i - (i!=0)
        for j in range(self.numNeurons[i]):
            curW.append(np.zeros(self.numNeurons[prev],).tolist())
        curB = np.zeros(self.numNeurons[prev],).tolist()
        outW.append(curW)
        outB.append(curB)
    return (outW, outB)
  
  def stochasticGradDesc(self, x_train, y_train, maxIterations=1000, learningRate=1.0, batchSize=5):
    if(len(self.weights) == 0):
        print("Initialising parameters")
        self.initialiseParams(len(x_train[0])*len(x_train[0]))
    for t in range(maxIterations):
      learningRate = learningRate/(1+(self.weightDecay*t))
      for i in range(batchSize):
        sample = np.random.randint(3*len(x_train)/4)
        self.forwardPropagate(x_train[sample])
        self.backwardPropagate(x_train[sample], y_train[sample])
      self.updateWeights(learningRate)
      self.updateBiases(learningRate)

  def momentumGradDesc(self, x_train, y_train, maxIterations=10, learningRate=0.1, batchSize=10, gamma=0.9):
    if(len(self.weights) == 0):
        print("Initialising parameters")
        self.initialiseParams(len(x_train[0])*len(x_train[0]))
    prev_v_w, prev_v_b = self.zeroMatrices()
    for t in range(maxIterations):
      learningRate = learningRate/(1+(self.weightDecay*t))
      for i in range(batchSize):
        sample = np.random.randint(3*len(x_train)/4)
        self.forwardPropagate(x_train[sample])
        self.backwardPropagate(x_train[sample], y_train[sample])
      v_w, v_b = self.updateVelocityWB(prev_v_w, prev_v_b, gamma, learningRate)
      self.updateWeights(learningRate,v_w)
      self.updateBiases(learningRate,v_b)
      prev_w_b = v_w
      prev_v_b = v_b
      del v_w
      del v_b

  def nesterovUpdateWeights(self, v_w):
    for i in range(self.numLayers):
      for j in range(self.numNeurons[i]):
        for k in range(len(self.weights[i][j])):
          self.weights[i][j][k] = self.weights[i][j][k] - v_w[i][j][k]

  def nesterovUpdateBiases(self, v_b):
    for i in range(self.numLayers):
      for j in range(self.numNeurons[i]):
        self.biases[i][j] = self.biases[i][j] - v_b[i][j]

  def nesterovAcceleratedGradDesc(self, x_train, y_train, maxIterations=10, learningRate=0.1, batchSize=1, gamma=0.9):
    if(len(self.weights) == 0):
      print("Initialising parameters")
      self.initialiseParams(len(x_train[0])*len(x_train[0]))
    prev_v_w, prev_v_b = self.zeroMatrices()
    for t in range(maxIterations):
      learningRate = learningRate/(1+(self.weightDecay*t))
      #Partial update
      v_w, v_b = self.updateVelocityWB(prev_v_w, prev_v_b, gamma, 0)
      for i in range(batchSize):
        sample = np.random.randint(3*len(x_train)/4)
        self.forwardPropagate(x_train[sample])
        originalWeights = self.weights.copy()
        originalBiases = self.biases.copy()
        self.nesterovUpdateWeights(v_w)
        self.nesterovUpdateBiases(v_b)
        self.backwardPropagate(x_train[sample], y_train[sample])
        self.weights = originalWeights.copy()
        self.biases = originalBiases.copy()
        del originalWeights
        del originalBiases
      #Full update
      v_w, v_b = self.updateVelocityWB(prev_v_w, prev_v_b, gamma, learningRate)
      self.updateWeights(learningRate,v_w)
      self.updateBiases(learningRate,v_b)
      prev_w_b = v_w
      prev_v_b = v_b
      del v_w
      del v_b

  def updateVelocityWBrmsprop(self, v_w, v_b, beta1, power=2):
    for i in range(self.numLayers):
        for j in range(self.numNeurons[i]):
            v_b[i][j] = beta1*v_b[i][j]
            for k in range(len(v_w[i][j])):
                v_w[i][j][k] = beta1*v_w[i][j][k]
                if len(self.gradW) != 0:
                    v_w[i][j][k] = v_w[i][j][k] + (1-beta1)*(self.gradW[i][j][k]**power)
            if len(self.gradB) != 0:
                v_b[i][j] = v_b[i][j] + (1-beta1)*(self.gradB[i][j]**power)
    return (v_w, v_b)

  def RMSupdateWeights(self,eta,v,eps):
    for i in range(self.numLayers):
      for j in range(self.numNeurons[i]):
        for k in range(len(self.gradW[i][j])):
          self.weights[i][j][k] = self.weights[i][j][k] - (eta/np.sqrt(abs(v[i][j][k] + eps)))*self.gradW[i][j][k]

  def RMSupdateBiases(self,eta,v, eps):
    for i in range(self.numLayers):
      for j in range(self.numNeurons[i]):
        self.biases[i][j] = self.biases[i][j] - (eta/np.sqrt(abs(v[i][j] + eps)))*self.gradB[i][j]

  def rmsprop(self, x_train, y_train, maxIterations=10, learningRate=0.1, batchSize=1, eps = 1e-8, beta1=0.9):
    if(len(self.weights) == 0):
      print("Initialising parameters")
      self.initialiseParams(len(x_train[0])*len(x_train[0]))
    v_w, v_b = self.zeroMatrices()
    for t in range(maxIterations):
      learningRate = learningRate/(1+(self.weightDecay*t))
      for i in range(batchSize):
        sample = np.random.randint(3*len(x_train)/4)
        self.forwardPropagate(x_train[sample])
        self.backwardPropagate(x_train[sample], y_train[sample])
      v_w, v_b = self.updateVelocityWBrmsprop(v_w, v_b, beta1)
      self.RMSupdateWeights(learningRate, v_w, eps)
      self.RMSupdateBiases(learningRate, v_b, eps)

  def _ADAMhelper(self, v_w, v_b, beta1, t):
    new_v_w, new_v_b = self.zeroMatrices()
    denom = 1 - (beta1**t)
    if denom != 0:
      for i in range(self.numLayers):
        for j in range(self.numNeurons[i]):
          new_v_b[i][j] = v_b[i][j]/denom
          for k in range(len(v_w[i][j])):
            new_v_w[i][j][k] = v_w[i][j][k]/denom
    return (new_v_w, new_v_b)

  def ADAMupdateWeights(self, eta, m, v, eps):
    for i in range(self.numLayers):
      for j in range(self.numNeurons[i]):
        for k in range(len(m[i][j])):
          self.weights[i][j][k] = self.weights[i][j][k] - ((eta/np.sqrt(abs(v[i][j][k] + eps)))*m[i][j][k])

  def ADAMupdateBiases(self, eta, m, v, eps):
    for i in range(self.numLayers):
      for j in range(self.numNeurons[i]):
        self.biases[i][j] = self.biases[i][j] - ((eta/np.sqrt(abs(v[i][j] + eps)))*m[i][j])

  def adam(self, x_train, y_train, maxIterations=10, learningRate=0.1, batchSize=1, eps = 1e-8, beta1=0.9, beta2=0.999):
    if(len(self.weights) == 0):
      print("Initialising parameters")
      self.initialiseParams(len(x_train[0])*len(x_train[0]))
    m_w, m_b = self.zeroMatrices()
    v_w, v_b = self.zeroMatrices()
    for t in range(maxIterations):
      learningRate = learningRate/(1+(self.weightDecay*t))
      for i in range(batchSize):
        sample = np.random.randint(3*len(x_train)/4)
        self.forwardPropagate(x_train[sample])
        self.backwardPropagate(x_train[sample], y_train[sample])
      m_w, m_b = self.updateVelocityWBrmsprop(m_w, m_b, beta1,1)
      v_w, v_b = self.updateVelocityWBrmsprop(v_w, v_b, beta2)
      m_w_hat, m_b_hat = self._ADAMhelper(m_w, m_b, beta1, t)
      v_w_hat, v_b_hat = self._ADAMhelper(v_w, v_b, beta2, t)
      self.ADAMupdateWeights(learningRate, m_w_hat, v_w_hat, eps)
      self.ADAMupdateBiases(learningRate, m_b_hat, v_b_hat, eps)

  def nadam(self, x_train, y_train, maxIterations=10, learningRate=0.1, batchSize=1, eps = 1e-8, beta1=0.9, beta2=0.999):
    if(len(self.weights) == 0):
      print("Initialising parameters")
      self.initialiseParams(len(x_train[0])*len(x_train[0]))
    m_w, m_b = self.zeroMatrices()
    v_w, v_b = self.zeroMatrices()
    for t in range(maxIterations):
      learningRate = learningRate/(1+(self.weightDecay*t))
      #Partial update
      v_w, v_b = self.updateVelocityWB(v_w, v_b, beta1, 0)
      for i in range(batchSize):
        sample = np.random.randint(3*len(x_train)/4)
        self.forwardPropagate(x_train[sample])
        originalWeights = self.weights.copy()
        originalBiases = self.biases.copy()
        self.nesterovUpdateWeights(v_w)
        self.nesterovUpdateBiases(v_b)
        self.backwardPropagate(x_train[sample], y_train[sample])
        self.weights = originalWeights.copy()
        self.biases = originalBiases.copy()
        del originalWeights
        del originalBiases
      #Full update
      m_w, m_b = self.updateVelocityWBrmsprop(m_w, m_b, beta1,1)
      v_w, v_b = self.updateVelocityWBrmsprop(v_w, v_b, beta2)
      m_w_hat, m_b_hat = self._ADAMhelper(m_w, m_b, beta1, t)
      v_w_hat, v_b_hat = self._ADAMhelper(v_w, v_b, beta2, t)
      self.ADAMupdateWeights(learningRate, m_w_hat, v_w_hat, eps)
      self.ADAMupdateBiases(learningRate, m_b_hat, v_b_hat, eps)
