import sys
import json
import time
import socket
import numpy as np
import pandas as pd


class Client:
    def __init__(self):
        self.client_id = sys.argv[1]
        self.host = "127.0.0.1"
        self.port = int(sys.argv[2])
        self.method = int(sys.argv[3])
        self.batch = 64
        self.learning_rate = 0.01
        self.epochs = 100
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.train_samples = 0
        self.test_samples = 0
        self.w = None

    def load_data(self):
        # load data set
        train_set = pd.read_csv(
            f"FLData/calhousing_train_{self.client_id}.csv")

        test_set = pd.read_csv(
            f"FLData/calhousing_test_{self.client_id}.csv")

        # split into features(x) and a target(y)
        x_train = train_set.iloc[:, :-1].values
        y_train = train_set.iloc[:, -1].values

        x_test = test_set.iloc[:, :-1].values
        y_test = test_set.iloc[:, -1].values

        train_samples, test_samples = len(y_train), len(y_test)

        # store values
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_samples = train_samples
        self.test_samples = test_samples

    def normalise_data(self):
        # numeric value normalisation due to different numerical scales on each feature
        # if not normalise, overflow or underflow occurs during GD
        self.x_train = (self.x_train - np.mean(self.x_train,
                        axis=0)) / np.std(self.x_train, axis=0)

        # add extra one's at the beginning of x_train
        train_ones = np.ones((self.x_train.shape[0], 1))
        self.x_train = np.concatenate((train_ones, self.x_train), axis=1)

        test_ones = np.ones((self.x_test.shape[0], 1))
        self.x_test = np.concatenate((test_ones, self.x_test), axis=1)

    def register_to_server(self):
        """
        Register the current client node to the server
        """
        try:
            # load data
            self.load_data()

            # normalise data
            self.normalise_data()

            # registration data
            data = {"client_id": self.client_id,
                    "client_port": self.port, "train_samples": self.train_samples}

            # encode the registration information
            encoded = json.dumps(data).encode("utf-8")

            # open a client TCP socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:

                # connect to the server (port: 6000)
                client.connect((self.host, 6000))

                # send encoded info
                client.send(encoded)

                # close the socket
                client.close()

        except Exception as e:
            print(f"Error: {e}")

    def test(self):
        x = self.x_test
        y = self.y_test
        w = self.w

        N = len(y)

        prediction = x.dot(w)
        error = prediction - y

        loss = 1/(2*N) * np.dot(error.T, error)

        return loss

    def gradient_descent(self):

        x = self.x_train
        y = self.y_train
        w = self.w
        epochs = self.epochs
        learning_rate = self.learning_rate

        past_loss = []
        past_w = [w]

        N = len(y)

        for _ in range(epochs):
            prediction = x.dot(w)
            error = prediction - y

            loss = 1/(2*N) * np.dot(error.T, error)
            past_loss.append(loss)

            mse = (1 / N) * np.dot(x.T, error)
            w = w - learning_rate * mse
            past_w.append(w)

        return past_w, past_loss

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:

            # open client socket
            client_socket.bind((self.host, self.port))
            client_socket.listen(5)

            while True:

                try:
                    print(f"I am {self.client_id}")

                    server_socket, addr = client_socket.accept()

                    # receive model from server
                    recv = server_socket.recv(1024)
                    gloabl_model = np.frombuffer(
                        recv, dtype=np.float64)
                    self.w = gloabl_model
                    print("Received new global model")

                    test_loss = self.test()
                    print(f"Testing MSE: {test_loss}")

                    # run gradient descent
                    print("Local training...")
                    past_w, past_loss = self.gradient_descent()
                    print(f"Training MSE: {past_loss[-1]}")

                    with open(f"Logs/{self.client_id}_log.txt", "a") as logfile:
                        log = f"Testing MSE: {test_loss} Training MSE: {past_loss[-1]}\n\n"
                        logfile.write(log)

                    # send local model to server
                    local_model = past_w[-1]
                    data = local_model.tobytes()
                    server_socket.send(data)
                    print("Sending new local model\n")

                    server_socket.close()

                except Exception as e:
                    print(f"Error: {e}")


def main():

    # initialisation of client
    client = Client()

    # register to server
    client.register_to_server()

    # run algorithm
    client.run()


if __name__ == "__main__":
    main()
