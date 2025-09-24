import matplotlib.pyplot as plt

xValues = [11, 14.66, 22, 33, 44]
yValues = [4.9724, 4.8142, 4.4954, 4.0182, 3.5403]

xValuesRed = [11,44]
yValuesRed = [4.9724, 3.5403]

def f(x):
    return -0.043397*x+5.4498


plt.plot(xValues, yValues)

for x in xValues:
    print(f(x))

plt.show()