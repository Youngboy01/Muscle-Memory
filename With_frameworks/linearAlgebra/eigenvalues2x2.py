import math
import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	# m = len(matrix)
	# n = len(matrix[0])
	# trace = 0
	# for i in range(m):
	# 	trace += matrix[i][i]
	# det=0
	# det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
	# egval1 = (trace + math.sqrt(trace**2 - 4*det))/(2) 
	# egval2 = (trace - math.sqrt(trace**2 - 4*det))/(2) 
	# eigenvalues = sorted([egval1,egval2],reverse=True)
	a = np.asarray(matrix)
	eigenvalues = np.linalg.eigvals(a)
	return eigenvalues