import re
import pickle


def get_preprocessed_binary():
    with open('./data/train_200K_back.pkl', 'rb') as f:
        binary_string = pickle.load(f)

    pattern = "000110111100111010010000100000000110111100111010010000100000000110111100111010010000100000"

    new_binary_string = binary_string.replace(pattern, "\n")
    with open('./data/train_200K_back_preprocessed.txt', 'w') as f:
        f.write(new_binary_string)
    
    new_binary_string = new_binary_string.split("\n")
    new_binary_string = [line[:-1] for line in new_binary_string]
    
    return new_binary_string


if __name__ == "__main__":
    get_preprocessed_binary()