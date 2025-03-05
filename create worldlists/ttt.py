t = [[],[]]

with open('spanish.txt', 'r', encoding='utf-8') as file:
    cont = True
    for i, line in enumerate(file):
        if i % 2 == 0 and len(line) > 2:
            if line.strip().lower() in t[0] or ('el ' + line.strip()).lower() in t[0] or ('la ' + line.strip()).lower() in t[0] or ('un ' + line.strip()).lower() in t[0] or ('una ' + line.strip()).lower() in t[0] or (line[2:].strip().lower() in t[0] and line.strip()[:2].lower() in ('el','la','un','una')):
                cont = False
                continue
            else:
                cont = True
        elif i % 2 == 0:
            if line.strip().lower() in t[0] or ('el ' + line.strip()).lower() in t[0] or ('la ' + line.strip()).lower() in t[0] or ('un ' + line.strip()).lower() in t[0] or ('una ' + line.strip()).lower() in t[0]:
                cont = False
                continue
            else:
                cont = True
        
        if cont:
            t[i%2].append(line.strip().lower())


temp = 0

print(len(t[0]))
print(len(t[1]))

file1 = open('firstcolumn.txt', 'w', encoding='utf-8')
file2 = open('secondcolumn.txt', 'w', encoding='utf-8')
#file = open('temp.txt', 'w', encoding='utf-8')

for i in range(len(t[0])):
    if t[0][i] != 'Ignorowane':
        file1.write(t[0][i])
        file1.write('\n')
        file2.write(t[1][i])
        file2.write('\n')
        temp += 1

print(temp)