import os

pathname = './Logs'

# Logs folder exists
if os.path.exists(pathname):

    # remove existing old log files
    for filename in os.listdir(pathname):
        file_path = os.path.join(pathname, filename)
        try:
            os.remove(file_path)  # Remove the file or link
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# not exist
else:
    # create Logs folder
    os.makedirs(pathname)
