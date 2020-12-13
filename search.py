import os

def search():
    path_tmp = os.getcwd()

    path_to_prj = ""
    for i in path_tmp.split("/"):
        path_to_prj += i + "/"
        if i == "parking_search":
            break
    print("path to prj is {}".format(path_to_prj))
    return path_to_prj
