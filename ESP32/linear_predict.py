def f(x, a, b):
    return a * x + b

a = 40.97
b = -283.38

print("Linear regression")
for i in [10, 50, 100, 200, 300]:
    print(f"Surface {i} => Loyer {f(i, a, b):.0f}")
    
def f2(x, a, b, c):
    return a * x ** 2 + b * x + c

a = 0.07389
b = 20.64
c = 0

print("Polynomial 2 regression")
for i in [10, 50, 100, 200, 300]:
    print(f"Surface {i} => Loyer {f2(i, a, b, c):.0f}")