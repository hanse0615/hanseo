import re
import pickle

with open('./data/train_200K_back.pkl', 'rb') as f:
    binary_string = pickle.load(f)

pattern = "000110111100111010010000100000000110111100111010010000100000000110111100111010010000100000"
result = []

matches = re.findall(pattern, binary_string)
for match in matches:
    result.append(match)

with open('./data/preprocessed_train_200K_back.txt', 'w') as f:
    for line in result:
        f.write(line + '\n')

print(len(result))