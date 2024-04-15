import sys
import json
import time
import socket
import torch
import numpy as np


class Client:
    def __init__(self):
        self.client_id = sys.argv[1]
        self.host = "127.0.0.1"
        self.port = int(sys.argv[2])
        self.method = int(sys.argv[3])
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.train_samples = 0
        self.test_samples = 0

    def load_data(self):
        # load data set
        train_set = np.genfromtxt(
            f"FLData/calhousing_train_{self.client_id}.csv", delimiter=',', skip_header=1)

        test_set = np.genfromtxt(
            f"FLData/calhousing_test_{self.client_id}.csv", delimiter=',', skip_header=1)

        # split into features(x) and a target(y)
        x_train = train_set[:, :-1]
        y_train = train_set[:, -1]
        x_test = test_set[:, :-1]
        y_test = test_set[:, -1]

        train_samples, test_samples = len(y_train), len(y_test)

        # store values
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_samples = train_samples
        self.test_samples = test_samples

    def register_to_server(self):
        """
        Register the current client node to the server
        """
        try:
            # load data
            self.load_data()

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

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:

            # open client socket
            client_socket.bind((self.host, self.port))
            client_socket.listen(5)

            while True:

                try:
                    server_socket, addr = client_socket.accept()

                    recv = b'' + server_socket.recv(1024)
                    decoded = json.loads(recv.decode("utf-8"))

                    server_model = decoded["model"]
                    print(f"receive {server_model} model")

                    print("learning ...")
                    time.sleep(5)

                    client_id = self.client_id
                    print(f"send {client_id} model\n")
                    data = {"model": client_id}
                    encoded = json.dumps(data).encode("utf-8")

                    server_socket.send(encoded)

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
