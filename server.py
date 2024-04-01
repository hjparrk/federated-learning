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


def register_client(conn, clients):

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

    init_phase = True

    while True:

        if init_phase:
            # create a TCP socket server
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind((host, port))
            server.listen(5)

            # first client tries to register
            try:
                conn, addr = server.accept()
                threading.Thread(target=register_client,
                                 args=(conn, clients)).start()
            except Exception as e:
                print("Error: {e}")

            # wait 30 seconds for other clients to register
            start_time = time.time()
            while time.time() - start_time < 30:
                server.settimeout(30 - (time.time() - start_time))
                try:
                    conn, addr = server.accept()
                    threading.Thread(target=register_client,
                                     args=(conn, clients)).start()
                except socket.timeout:
                    server.close()
                    init_phase = False
                    break

        else:
            print(clients)
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
