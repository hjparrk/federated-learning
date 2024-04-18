import sys
import json
import socket
import numpy as np
import pandas as pd


class Client:

    def __init__(self, batch_size, epochs, learning_rate):
        self.host = "127.0.0.1"
        self.client_id = sys.argv[1]
        self.port = int(sys.argv[2])
        self.method = int(sys.argv[3])
        assert self.method in [0, 1]  # full or mini-batch

        self.batch_size = batch_size  # for mini batch optimisation
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.w = None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

    def load_data(self):
        # load data set
        prefix = "FLData/calhousing"
        train_set = pd.read_csv(f"{prefix}_train_{self.client_id}.csv")
        test_set = pd.read_csv(f"{prefix}_test_{self.client_id}.csv")
        self.columns = train_set.columns

        # split into features(x) and a target(y)
        self.x_train = train_set.iloc[:, :-1].values
        self.y_train = train_set.iloc[:, -1].values

        self.x_test = test_set.iloc[:, :-1].values
        self.y_test = test_set.iloc[:, -1].values

        self.x_std = np.std(self.x_train, axis=0)
        self.x_mean = np.mean(self.x_train, axis=0)

    def normalise_data(self):
        # Normalise by standardising the training data set, which scales down columns with
        # bigger values such as population and scales up columns with relatively smaller values.
        # This allows us to speed up training process (gradient descent) by having large enough
        # learning rate without overflowing or underflowing any of the columns.

        self.x_train = (self.x_train - self.x_mean) / self.x_std

        train_ones = np.ones((self.x_train.shape[0], 1))
        self.x_train = np.concatenate((train_ones, self.x_train), axis=1)

        test_ones = np.ones((self.x_test.shape[0], 1))
        self.x_test = np.concatenate((test_ones, self.x_test), axis=1)

    def register_to_server(self):
        try:
            self.load_data()
            self.normalise_data()

            # registration data
            data = {
                "client_id": self.client_id,
                "client_port": self.port,
                "train_samples": len(self.y_train),
            }

            # encode the registration information
            encoded = json.dumps(data).encode("utf-8")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.connect((self.host, 6000))
                client.send(encoded)
                client.close()

        except Exception as e:
            print(f"Error: {e}")

    def test(self):
        # denormalise weights(coefficients) to fit the original scale
        W = self.w / [1, *self.x_std]
        W[0] = self.w[0] - sum(self.w[1:] * self.x_mean / self.x_std)
        # self.print_weight(W)

        prediction = self.x_test @ W
        error = prediction - self.y_test
        mse = (error.T @ error) / len(self.y_test)
        return mse

    def gradient_descent(self):
        MSE_record = []
        N = len(self.y_train)

        for _ in range(self.epochs):
            # full-batch
            if self.method == 0:
                error = (self.x_train @ self.w) - self.y_train
                adjustment = (self.x_train.T @ error) / N
                mse = (error.T @ error) / len(self.y_train)

            # mini-batch
            else:
                r = np.random.randint(0, N, size=self.batch_size)
                x_subset, y_subset = self.x_train[r], self.y_train[r]
                error = (x_subset @ self.w) - y_subset
                adjustment = (x_subset.T @ error) / self.batch_size
                mse = (error.T @ error) / len(y_subset)

            self.w = self.w - (adjustment * self.learning_rate)

            MSE_record.append(mse)

        return MSE_record

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:

            # open client socket
            client_socket.bind((self.host, self.port))
            client_socket.listen(5)
            print(f"I am {self.client_id}")

            # clear log file for whole new training session
            with open(f"Logs/{self.client_id}_log.txt", "w") as logfile:
                logfile.write("Testing MSE,Training MSE\n")

            while True:
                try:
                    # receive model from server
                    server_socket, _ = client_socket.accept()
                    recv = server_socket.recv(1024)

                    try:
                        gloabl_model = np.frombuffer(recv, dtype=np.float64)
                        self.w = gloabl_model
                        print("Received new global model")
                    except Exception:
                        # received out of structure message (likely server-closing message)
                        print("Detected server closing. Ending the client program")
                        break

                    test_set_mse = self.test()
                    print(f"Testing MSE: {test_set_mse}")

                    # run gradient descent
                    print("Local training...")
                    MSE_record = self.gradient_descent()
                    print(f"Training MSE: {MSE_record[-1]}")

                    with open(f"Logs/{self.client_id}_log.txt", "a") as logfile:
                        logfile.write(f"{test_set_mse},{MSE_record[-1]}\n")

                    # send local model to server
                    server_socket.send(self.w.tobytes())
                    print("Sending new local model\n")
                    server_socket.close()

                except Exception as e:
                    print(f"Error: {e}")

    def print_weight(self, weight):
        columns = ["y-intercept", *self.columns]
        for i in range(len(weight)):
            print(f"{columns[i]}: %.5f" % weight[i])


def main():
    np.random.seed(0)
    client = Client(batch_size=300, epochs=50, learning_rate=0.1)
    client.register_to_server()
    client.run()


if __name__ == "__main__":
    main()
