import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import time

def read_matrix_from_csv(filename, size):
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Calculate number of elements needed (just one matrix now)
    total_elements = size * size
    
    print(f"Reading file: {filename}")
    # Read only the required number of elements
    data = []
    count = 0
    
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            data.extend([float(val) for val in values])
            count += len(values)
            if count >= total_elements:
                break
    
    # Ensure we got enough data
    if len(data) < total_elements:
        raise ValueError(f"Not enough elements in file. Expected {total_elements}, got {len(data)}")
    
    # Only keep the exact number we need
    data = np.array(data[:total_elements])
    print(f"Data loaded, size: {data.size}")
    
    # Reshape into a matrix
    A = data.reshape(size, size)
    print(f"Successfully reshaped into matrix of size {size}x{size}")
    return A

def matrix_multiply_benchmark(filename, size=2):
    try:
        print(f"Starting benchmark with file: {filename}")
        A = read_matrix_from_csv(filename, size)
        
        print("Running warm-up multiplication...")
        _ = A @ A
        
        print("Running timed multiplication...")
        start = time.perf_counter()
        C = A @ A
        end = time.perf_counter()
        
        operations = 2 * size * size * size
        time_taken = end - start
        gflops = (operations / time_taken) / 1e9
        
        print(f"Matrix size: {size}x{size}")
        print(f"Time taken: {time_taken:.4f} seconds")
        print(f"Performance: {gflops:.2f} GFLOPS/s")
        
        print("Saving result...")
        np.savetxt('multiply/numpyResult.csv', C, delimiter=',')
        
        return C
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    result = matrix_multiply_benchmark('multiply/2048x2048.csv')
