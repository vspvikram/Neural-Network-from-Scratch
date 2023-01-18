import matplotlib.pyplot as plt
import numpy as np


# plotting the image from pixel indices and values
def plot_image(x, y, name):
    # plt.cla()
    x1_xsize = np.max(x[:, 0])
    x1_ysize = np.max(x[:, 1])

    image = np.zeros((x1_xsize, x1_ysize))
    for i, x in enumerate(x):
        image[x[0]-1, x[1]-1] = y[i][0]\

    f = plt.Figure()
    plt.xlabel("Pixels")
    plt.ylabel("Pixels")
    plt.title("Image: {}".format(name.split(".")[0]))
    plt.imshow(image)
    plt.savefig(name, format="png")


def plot_loss_curve(loss, name):
    plt.cla()
    plt.close()
    f2 = plt.Figure()
    plt.plot(loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(f"loss_curve_{name}.png")
    plt.show()

def predict_image(model, x, y, batch_size, name):
    y_hat = model.predict(x, batch_size=batch_size)
    
    print("The RMSE of the {} results: {}".format(name, rmse(y, y_hat)))
    y_hat = (y_hat * 255).astype(np.int16)
    plot_image(x, y_hat, name=f"{name}_pred_image.png")

def test_fun(X):
    return np.power(X, 2) + 1

def rmse(y_true, y_pred):
    output = np.sum(np.power(y_true - y_pred, 2))
    return np.sqrt(output)