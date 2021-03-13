import wandb
run = wandb.init(project="cs6910_assignment1", entity="rishanthrajendhran")
runs = [
    {
        "name": "5_5_128_0.5_0.0001_nadam_32_random_tanh",
        "test_acc": 0.1258,
        "test_loss": 34498.784738001894,
        "test_mse": 14.4475, 
        "valid_acc": 0.1335,
        "valid_loss": 20583.15743480236,
        "valid_mse": 14.01, 
    },
    {
        "name": "10_3_128_0_0.0001_momentum_64_random_sigmoid",
        "test_acc": 0.4836,
        "test_loss": 56111.169788829146,
        "test_mse": 4.1725, 
        "valid_acc": 0.4861666666666667,
        "valid_loss": 33064.48118814381,
        "valid_mse": 4.258666666666667, 
    }, 
    {
        "name": "5_5_32_0.5_0.0001_sgd_16_random_tanh",
        "test_acc": 0.1952,
        "test_loss": 36152.060580653335,
        "test_mse": 21.1266, 
        "valid_acc": 0.1925,
        "valid_loss": 21609.97438128073,
        "valid_mse": 21.118833333333335, 
    },
    {
        "name": "10_4_128_0.0005_0.0001_rmsprop_64_random_sigmoid",
        "test_acc": 0.3552,
        "test_loss": 52860.26295124708,
        "test_mse": 6.9387, 
        "valid_acc": 0.3581666666666667,
        "valid_loss": 31849.523257386296,
        "valid_mse": 6.913166666666666, 
    },
    {
        "name": "10_3_128_0.0005_0.0001_momentum_32_random_sigmoid",
        "test_acc": 0.3595,
        "test_loss": 57297.50592550861,
        "test_mse": 5.4654, 
        "valid_acc": 0.3675,
        "valid_loss": 33869.16644403018,
        "valid_mse": 5.519166666666667, 
    },
    {
        "name": "10_3_128_0_0.0001_momentum_64_xavier_tanh",
        "test_acc": 0.4337,
        "test_loss": 56420.62842148082,
        "test_mse": 5.1717, 
        "valid_acc": 0.43983333333333335,
        "valid_loss": 33672.38020987896,
        "valid_mse": 5.0615, 
    },
    {
        "name": "10_3_128_0_0.001_momentum_64_random_tanh",
        "test_acc": 0.3429,
        "test_loss": 55031.36254299177,
        "test_mse": 6.7976, 
        "valid_acc": 0.3416666666666667,
        "valid_loss": 32794.78907560154,
        "valid_mse": 6.830333333333333, 
    },
    {
        "name": "10_3_64_0_0.0001_adam_64_xavier_tanh",
        "test_acc": 0.3603,
        "test_loss": 50252.04220437231,
        "test_mse": 6.1517, 
        "valid_acc": 0.371,
        "valid_loss": 30181.358492172487,
        "valid_mse": 5.923333333333333, 
    },
    {
        "name": "10_3_64_0_0.0001_rmsprop_64_xavier_sigmoid",
        "test_acc": 0.3591, 
        "test_loss": 60732.83979198791,
        "test_mse": 6.8946, 
        "valid_acc": 0.36616666666666664,
        "valid_loss": 36121.8544708087,
        "valid_mse": 6.7251666666666665, 
    },
    {
        "name": "10_4_128_0_0.0001_sgd_64_xavier_sigmoid",
        "test_acc": 0.2343,
        "test_loss": 49009.765822670575,
        "test_mse": 6.2576, 
        "valid_acc": 0.2295,
        "valid_loss": 29512.92408056016,
        "valid_mse": 6.2855, 
    },
    {
        "name": "10_3_128_0.0005_0.0001_momentum_64_random_sigmoid",
        "test_acc": 0.4874,
        "test_loss": 58512.9887310171,
        "test_mse": 4.65, 
        "valid_acc": 0.4771666666666667,
        "valid_loss": 35113.18684126332,
        "valid_mse": 4.9085, 
    },
    {
        "name": "10_3_128_0_0.0001_nesterov_64_random_sigmoid",
        "test_acc": 0.27,
        "test_loss": 64330.26183687574,
        "test_mse": 6.4822, 
        "valid_acc": 0.27066666666666667,
        "valid_loss": 38565.40016481753,
        "valid_mse": 6.457833333333333, 
    },
    {
        "name": "5_4_32_0_0.0001_nadam_32_random_sigmoid",
        "test_acc": 0.1,
        "test_loss": 64642.57876689772,
        "test_mse": 28.5, 
        "valid_acc": 0.10616666666666667,
        "valid_loss": 38893.2668243516,
        "valid_mse": 28.8995, 
    },
    {
        "name": "10_5_128_0.5_0.0001_rmsprop_32_xavier_sigmoid",
        "test_acc": 0.1,
        "test_loss": 60310.78061735377,
        "test_mse": 28.5, 
        "valid_acc": 0.107,
        "valid_loss": 35968.0960513807,
        "valid_mse": 28.367166666666666, 
    }
]

import matplotlib.pyplot as plt

test_loss = []
test_mse = []
test_acc = []
for run in runs:
    test_loss.append(run["test_loss"])
    test_mse.append(run["test_mse"])
    test_acc.append(run["test_acc"])

plt.title("MSE vs Accuracy")
plt.xlabel("Test MSE")
plt.ylabel("Test Accuracy")
plt.scatter(test_mse, test_acc)
wandb.log({"mseVSacc":wandb.Image(plt)})
plt.show()
plt.title("Cross Entropy Loss vs Accuracy")
plt.xlabel("Test CEL")
plt.ylabel("Test Accuracy")
plt.scatter(test_loss, test_acc)
wandb.log({"lossVSacc":wandb.Image(plt)})
plt.show()
