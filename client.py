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


def load_data(client_id):
    # load data set
    train_set = np.genfromtxt(
        f"FLData/calhousing_train_{client_id}.csv", delimiter=',', skip_header=1)

    test_set = np.genfromtxt(
        f"FLData/calhousing_test_{client_id}.csv", delimiter=',', skip_header=1)

    # split into features(x) and a target(y)
    x_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]

    # cast variables to torch type
    x_train = torch.Tensor(x_train).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.float32)
    x_test = torch.Tensor(x_train).type(torch.float32)
    y_test = torch.Tensor(y_train).type(torch.float32)

    train_samples, test_samples = len(y_train), len(y_test)

    return x_train, y_train, x_test, y_test, train_samples, test_samples


def register_to_server(host, client_id, port, train_samples):
    """
    Register the current client node to the server
    """
    try:
        # registration data
        data = {"client_id": client_id,
                "client_port": port, "train_samples": train_samples}

        # encode the registration information
        encoded = json.dumps(data).encode("utf-8")

        # open a client TCP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:

            # connect to the server (port: 6000)
            client.connect((host, 6000))

            # send encoded info
            client.send(encoded)

            # close the socket
            client.close()

    except Exception as e:
        print(f"Error: {e}")


def main():

    # initialisation of client
    client = Client()

    # load data
    x_train, y_train, x_test, y_test, train_samples, test_samples = load_data(
        client.client_id)

    # register to the server
    register_to_server(client.host, client.client_id,
                       client.port, train_samples)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:

        # open client socket
        client_socket.bind((client.host, client.port))
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

                client_id = client.client_id
                print(f"send {client_id} model\n")
                data = {"model": client_id}
                encoded = json.dumps(data).encode("utf-8")

                server_socket.send(encoded)

                server_socket.close()

            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
