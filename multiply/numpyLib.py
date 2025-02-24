# Set environment variables before importing numpy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
import pandas as pd

def calculate_gflops(milliseconds, R1=2, C1=4, C2=2):
    seconds = milliseconds / 1000.0
    operations = 2.0 * R1 * C2 * C1  # 2 operations per multiply-add
    return (operations / seconds) / 1e9  # Convert to GFLOPS

# Read matrices from CSV
matrix1 = pd.read_csv("multiply/2048x2048.csv", header=None).values
matrix2 = pd.read_csv("multiply/2048x2048.csv", header=None).values

# Ensure matrices are int32 type
matrix1 = matrix1.astype(np.double)
matrix2 = matrix2.astype(np.double)

# Shape the matrices correctly
matrix1 = matrix1[:2, :4]  # R1xC1 (2x4)
matrix2 = matrix2[:4, :2]  # R2xC2 (4x2)


print(matrix1)
print(matrix2)

# Measure multiplication time
start_time = time.time()
#result = np.matmul(matrix1, matrix2)
result = matrix1 @ matrix2
end_time = time.time()

elapsed_ms = (end_time - start_time) * 1000
gflops = calculate_gflops(elapsed_ms)

print(f"NumPy Time = {elapsed_ms:.2f}ms")
print(f"NumPy Performance = {gflops:.6f} GFLOPS/s")

# Save result to CSV in the same format as C++
with open("multiply/numpy.csv", 'w') as f:
    # Convert the first row to string and join with commas
    first_row = ','.join(map(str, result[0]))
    f.write(first_row)
    
    # For remaining rows, add newline then values
    for row in result[1:]:
        f.write('\n' + ','.join(map(str, row)))
