import torch
import numpy as np
from sklearn import linear_model, tree
from sklearn.neural_network import MLPClassifier
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

loss = None
d_loss_arr = []
g_loss_arr = []


class DiscriminatorNet(torch.nn.Module):
    """
    the discriminator that contains 2 hidden layer
    this networks works to decrease loss and learn what the adversarial examples are
    """
    def __init__(self, sigmoid):
        super(DiscriminatorNet, self).__init__()
        n_features = 128
        n_out = 1
        if sigmoid:
            self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 256)

            )
            self.hidden1 = nn.Sequential(
                nn.Linear(256, 512)
            )
            # self.hidden2= nn.Sequential(
            #     nn.Linear(512, 512)
            # )
            self.out = nn.Sequential(
                torch.nn.Linear(256, n_out),
                nn.Sigmoid()
            )
        else:
            self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 256)
            )
            self.hidden1 = nn.Sequential(
                nn.Linear(256, 512)
            )
            # self.hidden2= nn.Sequential(
            #     nn.Linear(512, 512)
            # )
            self.out = nn.Sequential(
                torch.nn.Linear(512, n_out),
                torch.nn.Tanh()
            )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        # x= self.hidden2(x)
        x = self.out(x)
        return x


class GeneratorNet(torch.nn.Module):
    """
    a neural network with two hidden layers
    this network is used to create adversarial examples
    """
    def __init__(self, sigmoid):
        super(GeneratorNet, self).__init__()
        n_features = 148
        n_out = 128
        if sigmoid:
            self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 256),
            )
            self.hidden1 = nn.Sequential(
                nn.Linear(256, 512)
            )
            # self.hidden2= nn.Sequential(
            #     nn.Linear(512, 512)
            # )
            self.out = nn.Sequential(
                nn.Linear(512, n_out),
                torch.nn.Sigmoid()
            )
        else:
            self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 256)
            )
            self.hidden1 = nn.Sequential(
                nn.Linear(256, 512)
            )
            # self.hidden2= nn.Sequential(
            #     nn.Linear(512, 512)
            # )
            self.out = nn.Sequential(
                nn.Linear(512, n_out),
                torch.nn.Tanh()
            )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        # x = self.hidden2(x)
        x = self.out(x)
        return x


def real_data_target(size):
    """
    creates a tensor of the expected labels of real data
    :param size:
    :return:
    """
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(size):
    """
    creates a tensor of the expcted labels of fake data
    :param size:
    :return:
    """
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def train_discriminator(optimizer, real_data, real_data_labels, fake_data, fake_data_labels):
    """
    trains the discriminator or real data and fake data
    and computes a loss for each of them
    """
    global discriminator, generator
    optimizer.zero_grad()
    prediction_real = discriminator(real_data)
    error_real = nn.MSELoss()(prediction_real,real_data_labels)
    error_real.backward()
    prediction_fake = discriminator(fake_data)
    error_fake = nn.MSELoss()(prediction_fake, fake_data_labels)
    error_fake.backward()
    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data, fake_data_labels):
    """
    trains the gnerator giving the discriminator fake data and computes tje loss
    """
    global discriminator, loss, generator
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = nn.MSELoss()(prediction, fake_data_labels)
    error.backward()
    optimizer.step()
    return error


def load_data():
    """
    loads the data set
    :return:
    """
    data_set = "data/data1.npz"
    data = np.load(data_set)
    xmal, ymal, xben, yben = data['xmal'], data['ymal'], data['xben'], data['yben']
    return (xmal, ymal), (xben, yben)


def train(num_epochs, blackbox, generator, d_optimizer, g_optimizer, train_tpr, test_tpr, sigmoid, isFirst=True):
    """
    trains MalGAN by creating a blackbox and feeding it the data then teaching a generator to make fake examples
    :param num_epochs:
    :param blackbox:
    :param generator:
    :param d_optimizer:
    :param g_optimizer:
    :param train_tpr:
    :param test_tpr:
    :param sigmoid:
    :param isFirst:
    :return:
    """
    global g_loss_arr, d_loss_arr
    if isFirst:
        test_size = .2
    else:
        test_size = .5
    (mal, mal_label), (ben, ben_label) = load_data()
    x_train_mal, x_test_mal, y_train_mal, y_test_mal = train_test_split(mal, mal_label, test_size=test_size)
    x_train_ben, x_test_ben, y_train_ben, y_test_ben = train_test_split(ben, ben_label, test_size=test_size)

    blackbox_x_train_mal, blackbox_y_train_mal, blackbox_x_train_ben, blackbox_y_train_ben = \
        x_train_mal, y_train_mal, x_train_ben, y_train_ben

    if isFirst:
        blackbox.fit(np.concatenate([mal, ben]), np.concatenate([mal_label, ben_label]))

    ytrain_ben_blackbox = blackbox.predict(blackbox_x_train_ben)
    train_TPR = blackbox.score(blackbox_x_train_mal, blackbox_y_train_mal)
    test_TPR = blackbox.score(x_test_mal, y_test_mal)
    print(train_TPR, test_TPR)
    train_tpr.append(train_TPR)
    test_tpr.append(test_TPR)
    batch_size = 64

    for epoch in range(num_epochs):

        for step in range(x_train_mal.shape[0] // batch_size):
            d_loss_batches = []
            g_loss_batches = []

            #  generate batch of malware
            idm = np.random.randint(0, x_train_mal.shape[0], batch_size)
            if sigmoid:
                noise = np.random.uniform(0, 1, (batch_size, 20))
            else:
                noise = np.random.uniform(-1, 1, (batch_size, 20))
            xmal_batch = x_train_mal[idm]

            # generate batch of benign
            idb = np.random.randint(0, xmal_batch.shape[0], batch_size)
            xben_batch = x_train_ben[idb]
            yben_batch = ytrain_ben_blackbox[idb]

            # generate MALWARE examples
            combined = np.concatenate([xmal_batch, noise], axis=1)
            fake_mal_data = generator(torch.from_numpy(combined).float())

            # change the labels based on which activation function is being used
            if sigmoid:
                ymal_batch = blackbox.predict(np.ones(fake_mal_data.shape) * (np.asarray(fake_mal_data.detach()) > 0.5))
            else:
                ymal_batch = blackbox.predict(np.ones(fake_mal_data.shape) * (np.asarray(fake_mal_data.detach()) > 0))

            xben_batch = torch.from_numpy(xben_batch).float()
            yben_batch = torch.from_numpy(yben_batch).float()
            yben_batch = yben_batch.unsqueeze(1)
            ymal_batch = torch.from_numpy(ymal_batch).float()
            ymal_batch = ymal_batch.unsqueeze(1)

            # train discriminator
            d_loss, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, xben_batch, yben_batch,
                                                                   fake_mal_data, ymal_batch)

            d_loss_batches.append(d_loss.item())

            # train generator on noise
            g_loss = train_generator(g_optimizer, torch.from_numpy(xmal_batch).float(), yben_batch)
            g_loss_batches.append(g_loss.item())

        # add loss to array
        d_loss_arr.append(min(d_loss_batches))
        g_loss_arr.append(min(g_loss_batches))

        # train true positive rate
        if sigmoid:
            noise = np.random.uniform(0, 1, (x_train_mal.shape[0], 20))
        else:
            noise = np.random.uniform(-1, 1, (x_train_mal.shape[0], 20))

        combined = np.concatenate([x_train_mal, noise], axis=1)
        gen_examples = generator(torch.from_numpy(combined).float())

        if sigmoid:
            train_TPR = blackbox.score(np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0.5),
                                       y_train_mal)
        else:
            train_TPR = blackbox.score(np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0),
                                       y_train_mal)

        # test true positive rate
        if sigmoid:
            noise = np.random.uniform(0, 1, (x_test_mal.shape[0], 20))
        else:
            noise = np.random.uniform(-1, 1, (x_test_mal.shape[0], 20))

        combined = np.concatenate([x_test_mal, noise], axis=1)
        gen_examples = generator(torch.from_numpy(combined).float())

        if sigmoid:
            test_TPR = blackbox.score(np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0.5),
                                      y_test_mal)
        else:
            test_TPR = blackbox.score(np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0),
                                      y_test_mal)

        print(train_TPR, " ", test_TPR)

        train_tpr.append(train_TPR)
        test_tpr.append(test_TPR)


def retrain(blackbox, generator, sigmoid):
    """
    retrain the blackbox after the malgan has beeen trained
    :param blackbox:
    :param generator:
    :param sigmoid:
    :return:
    """
    (mal, mal_label), (ben, ben_label) = load_data()
    x_train_mal, x_test_mal, y_train_mal, y_test_mal = train_test_split(mal, mal_label, test_size=0.20)
    x_train_ben, x_test_ben, y_train_ben, y_test_ben = train_test_split(ben, ben_label, test_size=0.20)

    # Generate Train Adversarial Examples
    if sigmoid:
        noise = np.random.uniform(0, 1, (x_train_mal.shape[0], 20))
    else:
        noise = np.random.uniform(-1, 1, (x_train_mal.shape[0], 20))
    combined = np.concatenate([x_train_mal, noise], axis=1)
    gen_examples = generator(torch.from_numpy(combined).float())

    gen_examples_np =np.asarray(gen_examples.detach())

    blackbox.fit(np.concatenate([x_train_mal, x_train_ben, gen_examples_np]),
                 np.concatenate([y_train_mal, y_train_ben, np.zeros(gen_examples.shape[0])]))

    # training true positive rate
    train_TPR = blackbox.score(np.asarray(gen_examples.detach()), y_train_mal)

    # test true positive rate
    if sigmoid:
        noise = np.random.uniform(0, 1, (x_test_mal.shape[0], 20))
    else:
        noise = np.random.uniform(-1, 1, (x_test_mal.shape[0], 20))

    combined = np.concatenate([x_test_mal, noise], axis=1)
    gen_examples = generator(torch.from_numpy(combined).float())

    if sigmoid:
        gen_examples = np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0.5)
    else:
        gen_examples = np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0)

    test_TPR = blackbox.score(gen_examples, y_test_mal)

    print('\n---TPR after the black-box detector is retrained(Before Retraining MalGAN).')
    print('\nTrain_TPR: {0}, Test_TPR: {1}'.format(train_TPR, test_TPR))


def main():
    global discriminator, generator, loss
    sigmoid = True


    # initialize the discriminator and generator
    discriminator = DiscriminatorNet(sigmoid)
    generator = GeneratorNet(sigmoid)

    ## DIFFERENT BLACKBOX OPTIONS TO TEST
    ## COMMENT OUT THE ONES THAT ARE NOT BEING TESTED
    # blackbox = RandomForestClassifier(n_estimators=101, max_depth=10, random_state=1)
    # blackbox = LinearSVC()
    blackbox = linear_model.LogisticRegression()

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # arrays for plotting
    TRAIN_TPR = []
    TEST_TPR = []
    POST_TRAIN_TPR = []
    POST_TEST_TPR = []

    # train the gan on examples
    train(200, blackbox, generator, d_optimizer, g_optimizer, TRAIN_TPR, TEST_TPR, sigmoid, isFirst=True)

    # retrain the blackbox
    retrain(blackbox, generator, sigmoid)

    # run malgan again
    train(75, blackbox, generator, d_optimizer, g_optimizer, POST_TRAIN_TPR, POST_TEST_TPR, sigmoid, isFirst=False)

    # print the loss arrays
    print(d_loss_arr)
    print(g_loss_arr)

    print(TEST_TPR[len(TEST_TPR)-1])

    print(POST_TEST_TPR[len(POST_TEST_TPR)-1])
    # plot data
    plt.plot(range(len(d_loss_arr)), d_loss_arr, label="Discriminator loss", color="blue")
    plt.plot(range(len(g_loss_arr)), g_loss_arr, label="Generator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Min Loss Value per Epoch")
    plt.title("Graph of losses over time")
    plt.legend()
    plt.show()

    plt.plot(range(len(TRAIN_TPR)), TRAIN_TPR, label="Training TPR", color="blue")
    plt.plot(range(len(TEST_TPR)), TEST_TPR, label="Testing TPR", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("TPR rate")
    plt.legend()
    plt.title("True Positive Rate of RF Blackbox before retraining MalGAN")
    plt.show()

    plt.plot(range(len(POST_TRAIN_TPR)), POST_TRAIN_TPR, label="Training TPR", color="blue")
    plt.plot(range(len(POST_TEST_TPR)), POST_TEST_TPR, label="Testing TPR", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("TPR rate")
    plt.legend()
    plt.title("True Positive Rate of RF Blackbox after retraining MalGAN")
    plt.show()


if __name__ == '__main__':
    main()
