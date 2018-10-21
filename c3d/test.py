weights = {
                'wc1': 1,
                'wc2': 2,
                'wc3a': 3,
                'wc3b': 4
            }
bias = {
                'wc11': 1,
                'wc22': 2,
                'wc33a': 3,
                'wc33b': 4
            }

print(weights.values())
print(set(weights.values()))
a = set(list(weights.values()) + list(bias.values()))
print(a)