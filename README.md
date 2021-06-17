<h1>Neural Network From Scratch</h1>
<h3>Team Members</h3>
  <ol>
    <li><strong>Rishanth. R</strong> (CS18B044)</li> 
  </ol>
<h3>Contents of the files</h3>
  <ul>
  <li>
    <strong>neuralNetwork.py</strong>
    <p>
      Contains the class Neural Network with all the associated functions required to initialise a neural network, forward propagate a training sample through it, 
      do backward propagation to compute the gradients and to run several gradient descent algorithms.
    </p>
  </li>
  <li>
    <strong>metrics.py</strong>
    <p>
      Contains code to compute the following metrics:
      <ul>
        <li>Accuracy</li>
        <li>Cross Entropy Loss</li>
        <li>Mean Squared Error</li>
      </ul>
    </p>
  </li>
  <li>
    <strong>train.py</strong>
    <p>
      Contains code to set up a neural network based on specified hyperparamters, train it over the normalised train data by forward propagation and then run gradient 
      descent algorithms to tune the paramters.<br/>
      The test and validation data (last 10% of the train data) are then passed through the neural network to compute the metrics mentioned above.<br/>
      <h6>Note:</h6>
        Due to an unresolved 'Maximum Recursion Error' wandb error, the entropy for both the validation and test sets could not be logged into wandb. 
    </p>
  </li>
  <li>
    <strong>sweep.yaml</strong>
    <p>
      Contains the sweep configuration.<br/>
      The bayes strategy was chosen for the sweep over grid search (computationally expensive) and random search (might settle for local minima)<br/>
      Since the entropy losses could not be logged into wandb due to aforementioned errors, validation set accuracy is the objective here with a goal to maximize it.
    </p>
  </li>
  <li>
    <strong>bestModel.py</strong>
    <p>
        Using the hyperparamters that gave the highest accuracy during the sweep, we train a neural network and plot the confusion matrix for the neural network over the test data using pandas and seaborn.
    </p>
  </li>
  <li>
    <strong>report.py</strong>
    <p>
        Due to the unresolved wandb error 'Maximum recusion depth', we could not log the cross entropy losses into wandb. The entropy and mse for the 14 runs visualised in wandb were manually collected and stored in a list of dictionaries. This is used to plot the scatter plots:<br/>
        <ol>
        <li>Test set mean squared error vs test set accuracy</li>
        <li>Test set cross entropy loss vs test set accuracy</li>
        </ol>
    </p>
  </li>
 </ul>
<h3>Running the code</h3>
  <h5>Train a neural network</h5>
  <p>
    To train a neural network based on a specific set of hyperparamters, one has to first modify the hyperparameters_default dictionary, defined in train.py, 
    appropriately.<br/>
    After logging into your wand accound using 'wandb login', run the command 'python3 train.py' in the project directory terminal in the terminal.<br/>
    The code initialises a neural network with the specified hyperparameters, trains it over the normalised train data and then computes the aforementioned metrics
    over the test and validation sets.<br/>
    Due to an unresolved error (mentioned above), entropy losses could not be logged into wandb
  </p>
  <h5>Run a sweep</h5>
  <p>
    Run the command 'wandb sweep sweep.yaml' in the project directory in the terminal.<br/>
    Then run the command 'wandb agent %sweep-agent-generated-by-previous-command-here%' to start the sweep.
  </p>
</p>
