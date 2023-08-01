filename = '/data/omanma1928/server12/projects/mong/09_braille_bart/data/preprocess.txt'
lines = []
with open(filename, 'r') as f:
    line = f.readline()
    while line:
        lines.append(line.strip()+' 뷁뷁뷁뷁뷁\n')
        line = f.readline()


with open('./data/preprocess_backbackback.txt', 'w') as f:
    for line in lines:
        f.write(line)
        
        
        