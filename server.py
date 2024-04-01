import sys
import json
import time
import socket
import threading

import torch
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size=1):
        super(LinearRegressionModel, self).__init__()
        # Create a linear transformation to the incoming data
        self.linear = nn.Linear(input_size, 1)

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Apply linear transformation
        output = self.linear(x)
        return output.reshape(-1)


def handle_client(conn, clients):

    try:
        # receive and decode registration info
        recv = b'' + conn.recv(1024)
        decoded = json.loads(recv.decode("utf-8"))

        # register client
        clients.append(decoded)

        # close the connection
        conn.close()

    except Exception as e:
        print(f"Error: {e}")


def run_server(host, port, num_subsamples, clients):

    # create a TCP socket server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(num_subsamples)

    init_phase = True

    while True:
        try:
            # accept the connection from a client
            conn, addr = server.accept()

            if init_phase:
                # register client
                client_thread = threading.Thread(
                    target=handle_client, args=(conn, clients))
                client_thread.start()

        except Exception as e:
            print(f"Error: {e}")
            server.close()
            break


def main():

    # init
    clients = []
    host = "127.0.0.1"
    port = int(sys.argv[1])
    num_subsamples = 5 if int(sys.argv[2]) == 0 else int(sys.argv[2])

    # init model
    model = LinearRegressionModel()
    batch_size = 64
    learning_rate = 0.01
    num_glob_iters = 10  # No. of global rounds

    # create server
    run_server(host, port, num_subsamples, clients)


if __name__ == "__main__":
    main()
