import os

# GLOBAL VARIABLES
folder = "./results"


def clean_result_folder(path=""):
    directory = folder + path
    for the_file in os.listdir(directory):
        file_path = os.path.join(directory, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
