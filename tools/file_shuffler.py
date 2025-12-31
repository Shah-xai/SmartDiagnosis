import random

def shuffle_lst(input_path:str, output_path:str=None,seed=42):
    with open(input_path,"r") as f:
        lines = f.readlines()

    data = lines
    random.seed(seed)
    random.shuffle(data)
    if output_path is None:
        output_path=input_path
    with open(output_path,"w") as f:
            f.writelines(data)
