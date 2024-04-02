import sys
import json
import time
import select
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


class Server():
    def __init__(self):
        # init
        self.clients = []
        self.host = "127.0.0.1"
        self.port = int(sys.argv[1])
        self.num_subsamples = 5 if int(sys.argv[2]) == 0 else int(sys.argv[2])
        self.server_model = LinearRegressionModel(8)
        self.batch_size = 64
        self.learning_rate = 0.01
        self.num_glob_iters = 10  # No. of global rounds
        self.total_train_samples = 0


def calculate_total_train_samples(clients):
    # calculate total_train_samples
    total_train_samples = 0
    for client in clients:
        total_train_samples += client["train_samples"]

    return total_train_samples


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


def run_init_phase(host, port, clients):
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
            break

    server.close()


def broadcast_model(server: Server, client):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:

        # connect to client
        host = server.host
        port = int(client["client_port"])
        server_socket.connect((host, port))

        data = {"model": "server"}
        encoded = json.dumps(data).encode("utf-8")

        client_id = client["client_id"]
        print(f"send model to {client_id}")
        # send server model
        server_socket.send(encoded)

        recv = b'' + server_socket.recv(1024)
        decoded = json.loads(recv.decode("utf-8"))

        local_model = decoded["model"]
        print(f"receive {local_model} model")

        print()


def run_epoch(server: Server):
    for client in server.clients:
        broadcast_model(server, client)


def run_server(server: Server):

    # run non-blocking server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server.host, server.port))
    server_socket.listen(20)
    server_socket.setblocking(False)

    inputs = [server_socket]

    for i in range(10):
        print(f"epoch {i}")
        # broadcast server model, receive local model, then aggregate
        run_epoch(server)

        # check for new client registration
        while True:

            # monitor readable sockets
            readable, writable, errored = select.select(inputs, [], [], 0)

            # break if there are no new socket connections
            if not readable:
                break

            for s in readable:

                if s is server_socket:

                    # connection from a new client
                    conn, address = s.accept()
                    conn.setblocking(False)
                    inputs.append(conn)

                else:
                    # receive registration data from the new client
                    recv = conn.recv(1024)

                    # data exists
                    if recv:
                        byted_recv = b'' + recv
                        decoded = json.loads(byted_recv.decode("utf-8"))

                        # register new client
                        server.clients.append(decoded)
                        server.total_train_samples = calculate_total_train_samples(
                            server.clients)

                        # close connection
                        inputs.remove(s)
                        s.close()

                    # no data received
                    else:
                        # close connection
                        inputs.remove(s)
                        s.close()


def main():

    # initialisation of server
    server = Server()

    # initial clients registration phase
    run_init_phase(server.host, server.port, server.clients)
    server.total_train_samples = calculate_total_train_samples(server.clients)

    # run algorithm
    run_server(server)


if __name__ == "__main__":
    main()
