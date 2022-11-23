import os

def create_folder(new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)

create_folder("data")
create_folder("data_ENM")
create_folder("data_ENMimages")
create_folder("input_files")
create_folder("faces")