# List of prime numbers as int
with open('prime_numbers.txt') as f:
    lines = f.readlines()
prime_list = lines[0].split(",")
prime_list = [int(n) for n in prime_list]

n_input_features = 390
conv_length_1 = 0
conv_length_2 = 0
conv_length_3 = 0
conv_length_4 = 0
conv_length_5 = 0
conv_length_6 = 0
conv_length_7 = 0

for p in prime_list:
    if p <= n_input_features // 2:
        conv_length_1 += n_input_features - p + 1

for p in prime_list:
    if p <= conv_length_1 // 2:
        conv_length_2+= conv_length_1 - p + 1

for n in [1, 2]:
    conv_length_3 += conv_length_2 - n + 1

conv_length_4 = conv_length_3 * 25

for p in prime_list:
    if p <= (conv_length_4) // 2:
        conv_length_5 += conv_length_4 - p + 1

for p in prime_list:
    if p <= conv_length_5 // 2:
        conv_length_6 += conv_length_5 - p + 1

for n in [1, 2]:
    conv_length_7 += conv_length_6 - n + 1

print('Conv Length 1: ', conv_length_1)
print('Conv Length 2: ', conv_length_2)
print('Conv Length 3: ', conv_length_3)
print('Conv Length 4: ', conv_length_4)
print('Conv Length 5: ', conv_length_5)
print('Conv Length 6: ', conv_length_6)
print('Conv Length 7: ', conv_length_7)