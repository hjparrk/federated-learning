# COMP3221 A2 Federated Learning - Group 15

## Prerequisites

Related packages to be installed in your local machine to run the program.

### Environment

MacOS, WindowOS

### Installation

Use the package manager `pip` to install the following packages.

```bash
pip install numpy
pip install pandas
```

If module not found error occurs like this,

```
John-Doe:ROOT-DIRECTORY johndoe$ python3 serverr.py
Traceback (most recent call last):
  File "/Users/johndoe/COMP3221/serverr.py", line 3, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

Please try the instructions below.

Open terminal, and enter `which python`.
Then, it will print the path of python in your machine like this.

```
/usr/bin/python
```

Copy the path, and try the command below in the terminal

```
/usr/bin/python -m pip install <package>
```

## Getting Started

### Assumptions

-   Only one server exists
-   At least one client
-   data.csv files must be stored in the folder `FLData`
-   log.txt files are stored in `Logs` folder (Logs folder will be auto-generated if server program runs)

### Folder Structure

To run the program appropriately, the folder must follow the structure below.

```
Root directory
│   server.py
│   client.py
│   readme.md
└───FLData
│   │   calhousing_test_client1.csv
│   │       ...
│   │   calhousing_test_client5.csv
│   │   calhousing_train_client1.csv
│   │       ...
│   │   calhousing_train_client5.csv
└───Logs
│   │   client1_log.txt
│   │   client2_log.txt
│   │       ...
│   │   client5_log.txt
```

### How It works?

You should run **exactly one** main server first then **at least one** client using the terminal in your local machine. Once the main server started and the first client registered, the server waits for **30 seconds** for other clients for registration (a.k.a initial registration phase).

After the initial registration phase, the server broadcasts(distributes) randomly generated initial global model denoted as `w` to its registered clinets. Each client receives the global model, trains the model for a certain number of epochs using test set of local data, update the model and send it back to the main server. Then, the server subsamples the received models and aggregates them, and broadcasts the updated global model to its clients again. It repeats these steps for the specified number of global iterations.

While the algorithm runs, new clients can register to the server at any time and server will accept registration and broadcast the global model to them from the next global iteration.

During each epoch, every client saves logs of testing and training output to a seperate logging file in **Logs** folder and also prints it out to the terminal like:

```
I am client 1
Received new global model
Testing MSE: 0.6922826265272948
Local training...
Training MSE: 0.5650364525245951
Sending new local model
```

During each global iteration, the server does not create any logging file but prints out the process to the terminal like:

```
Global Iteration 10:
Total Number of clients: 5
Getting local model from client 1
Getting local model from client 2
Getting local model from client 5
Getting local model from client 4
Getting local model from client 3
Aggregating new global model
Broadcasting new global mode
```

### How to run?

-   #### Server

It requires `two` command-line arguments: **port number** and **number of subsampling**, for executiion as follows:

```
python COMP3221_FLServer.py <Port-Server> <Sub-Client>
```

where `<Port-Server>` must be 6000 and `0 ≤ <Sub-Client> ≤ 5`.

-   #### Client

It requires `three` command-line arguments: **unique id**, **port number** and **optimizer method**, for executiion as follows:

```
python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>
```

where `<Client-id>` is a type of string, `<Port-Client>` is a port for each client and `Opt-Method` is the indication of optimizer method.

> #### restrictions
>
> -   Port numbers are assigned starting at 6001 for client1 and increment by one for each subsequent client, up to 6005 for client5.
> -   The optimization method used for local model training. A value of 0 selects Gradient Descent (GD), and a value of 1 selects Mini-Batch GD.

### Expected Output

If you re-run the program, the existing log data saved in the log file will be removed and new log data will be overwritten.

-   #### Logs/client1_log.txt

```
Testing MSE,Training MSE
4.403923718962869,0.8545464170197506
0.6922826265272948,0.5650364525245951
0.5757542720561656,0.5673026058898107
0.5196754633625289,0.6288565058272481
0.4935628428129273,0.4705730556132602
0.47896182979452173,0.6992004794204847
0.47475271573268285,0.5138587481023313
0.47092855172514725,0.5123981178592157
0.4688252726242364,0.3960182477391474
0.46795048679201673,0.5364627836116679
0.46946558723597087,0.5502550325394732
0.4687041919934277,0.500472558943524
0.46946232756323486,0.607317867690973
0.46939694437617907,0.5740862623301374
0.4689215248082723,0.457789571803383
0.4710696562175215,0.3980531920995816
0.46677853331784636,0.534504807788258
0.4689216138149234,0.48460728075177695
0.4678958485018793,0.5447561521327531
0.4730802242133825,0.5855830155302726
```
