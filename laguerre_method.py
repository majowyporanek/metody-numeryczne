import numpy as np
import numpy.polynomial.laguerre as nprooots

coef12a = [243, -486, 783, -990, 558, -28, -72, 16]
coef12b = [1, 1, 3, 2, -1, -3, -11, -8, -12, -4, -4]

num = complex(0,1);
coef12c = [1, num, -1, -num ,1]
print(coef12c)

result12a = nprooots.lagroots(coef12a)
result12b = nprooots.lagroots(coef12b)
result12c = nprooots.lagroots(coef12c)

print(result12a)
print(result12b)
print(result12c)