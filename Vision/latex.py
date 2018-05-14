output = '[0.65344362 0.76873734 0.85178933 0.91188386]'
output = output[1:-1].split(' ')
s = ''

for item in output:
    # print(item)
    temp = '& ' + str(int(1e4*float(item))/100)
    # print(temp)
    s = s + temp
print(s)

