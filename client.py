import sys
import json
import socket
import numpy as np
from torch.utils.data import DataLoader


def load_data(client_id):
    train_set = np.genfromtxt(
        f"FLData/calhousing_train_{client_id}.csv", delimiter=',', skip_header=1)
    test_set = np.genfromtxt(
        f"FLData/calhousing_test_{client_id}.csv", delimiter=',', skip_header=1)

    return train_set, test_set


def register_to_server(host, client_id, port, data_size):
    """
    Register the current client node to the server
    """
    try:
        # registration data
        # data = {"client_id": client_id,
        #         "client_port": port, "data_size": data_size}

        data = {"client_id": client_id}

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
    client_id = sys.argv[1]
    host = "127.0.0.1"
    port = int(sys.argv[2])
    method = int(sys.argv[3])

    train_set, test_set = load_data(client_id)

    # register_to_server(host, client_id, port, 300)


if __name__ == "__main__":
    main()
