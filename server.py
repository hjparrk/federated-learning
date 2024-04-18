import sys
import json
import time
import random
import select
import socket
import threading
import numpy as np


class Server:

    def __init__(self, global_iter):
        self.clients = []
        self.host = "127.0.0.1"
        self.port = int(sys.argv[1])
        self.num_subsamples = 5 if int(sys.argv[2]) == 0 else int(sys.argv[2])
        self.num_glob_iters = global_iter  # no. of global rounds
        self.total_train_samples = 0  # no. of training samples
        self.w = np.random.rand(9)  # global model
        self.locals = []

    def calculate_total_train_samples(self):
        # calculate total_train_samples
        self.total_train_samples = 0
        for client in self.clients:
            self.total_train_samples += client["train_samples"]

    def aggregate_local_models(self):

        # [0 0 0 0 0 0 0 0 0]
        aggregated_model = np.zeros(9, dtype=float)

        # lack of registered clients to subsample
        if len(self.clients) <= self.num_subsamples:
            # aggregate received local models
            for model in self.locals:
                aggregated_model += model

            # average models
            aggregated_model /= len(self.locals)

        # enough registered clients
        else:
            # subsample models
            selected_models = random.sample(self.locals, self.num_subsamples)
            for model in selected_models:
                aggregated_model += model

            # average models
            aggregated_model /= self.num_subsamples

        # replace old global model with newly aggregated model
        self.w = aggregated_model

        # empty the list for received local models
        self.locals.clear()

        print("Aggregating new global model")
        # print(f"Aggregation result: {self.w}")
        print()

    def register_client(self, conn):
        try:
            # receive and decode registration info
            recv = b"" + conn.recv(1024)
            decoded = json.loads(recv.decode("utf-8"))

            # register client
            self.clients.append(decoded)

            # close the connection
            conn.close()

        except Exception as e:
            print(f"Error: {e}")

    def run_init_phase(self):
        # create a TCP socket server
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(10)

        # print("Waiting for the first client to connect... (indefinite)")
        # first client tries to register
        try:
            conn, addr = server.accept()
            threading.Thread(target=self.register_client,
                             args=(conn, )).start()
        except Exception as e:
            print("Error: {e}")

        # print("Waiting for rest of the clients to connect... (30 seconds)")
        # wait 30 seconds for other clients to register
        start_time = time.time()
        while time.time() - start_time < 15:
            server.settimeout(30 - (time.time() - start_time))
            try:
                conn, addr = server.accept()
                threading.Thread(target=self.register_client,
                                 args=(conn, )).start()
            except socket.timeout:
                server.close()
                break

        server.close()

    def broadcast_model(self, client, send_end_msg):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:

            # connect to client
            host = self.host
            port = int(client["client_port"])
            server_socket.connect((host, port))

            if not send_end_msg:
                data = self.w.tobytes()
            else:
                data = "server-closing".encode()

            # send server model or the end msg
            server_socket.send(data)

            if not send_end_msg:
                # receive local model from the client
                recv = server_socket.recv(1024)
                local_model = np.frombuffer(recv, dtype=np.float64)
                self.locals.append(local_model)

                client_id = client["client_id"]
                print(f"Getting local model from {client_id}")

    def run_epoch(self, send_end_msg):

        threads = []

        # create threads
        for client in self.clients:
            th = threading.Thread(target=(self.broadcast_model),
                                  args=(client, send_end_msg))

            # store thread to list
            threads.append(th)

        # start threads
        for th in threads:
            th.start()

        # wait for all threads to finish the task
        for th in threads:
            th.join()

    def run(self):

        # run non-blocking server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(20)
        server_socket.setblocking(False)

        inputs = [server_socket]

        for i in range(self.num_glob_iters + 1):

            print(f"Global Iteration {i+1}")
            print(f"Total Number of clients: {len(self.clients)}")

            if i == self.num_glob_iters:  # end of server
                self.run_epoch(send_end_msg=True)
            else:
                print("Broadcasting new global model")
                # broadcast server model, receive local model, then aggregate
                self.run_epoch(send_end_msg=False)
                # aggregate local models and replace old global model with updated one
                self.aggregate_local_models()

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
                            byted_recv = b"" + recv
                            decoded = json.loads(byted_recv.decode("utf-8"))

                            # register new client
                            self.clients.append(decoded)
                            self.calculate_total_train_samples()

                            # close connection
                            inputs.remove(s)
                            s.close()

                        # no data received
                        else:
                            # close connection
                            inputs.remove(s)
                            s.close()


def main():

    np.random.seed(0)

    # initialisation of server
    server = Server(global_iter=20)

    # initial clients registration phase
    server.run_init_phase()
    server.calculate_total_train_samples()

    # run algorithm
    server.run()


if __name__ == "__main__":
    main()
