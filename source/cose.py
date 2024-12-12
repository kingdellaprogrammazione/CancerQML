#fancy way
from itertools import product
from pprint import pprint

# Inputs
now_encoding = ["amplitude", "angle"]
now_anstz = ["a", "b", "c", "d"]
now_numlayers = [1, 10, 10]
now_threshold = [0.1, 0.2, 0.3, 0.4]

# Generate combinations
combinations = list(product(now_anstz, now_numlayers, now_threshold, now_encoding))

# Add counter to each combination
lista_tot = [list(combo) + [idx + 1] for idx, combo in enumerate(combinations)]

# Print the result
print(lista_tot)

#not fancy
from pprint import pprint
now_encoding = ["amplitude","angle"]
now_anstz = ["a","b","c","d"]
now_numlayers = [1,10,10]
now_threshold = [0.1,0.2,0.3,0.4]

lista_tot = []
n = 0
for g in now_encoding :
    for i in now_anstz:
        for j in now_numlayers:
            for k in now_threshold:
                mini_lista = []
                mini_lista.append(i)
                mini_lista.append(j)
                mini_lista.append(k)
                mini_lista.append(g)
                n = n+1
                mini_lista.append(n)
                lista_tot.append(mini_lista)
pprint(lista_tot)
