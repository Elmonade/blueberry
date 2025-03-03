import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import time

# Set higher precision to avoid rounding
np.set_printoptions(
    precision=5,  # Use 5 digits of precision
    suppress=False,  # Don't suppress scientific notation
    floatmode="fixed",  # Use fixed number of digits (not rounded)
)

def truncate_formatter(x, precision=5):
    """Format number in scientific notation and truncate (not round) the mantissa"""
    # Convert to scientific notation with extra precision
    s = f"{x:.{precision+6}e}"
    # Split into mantissa and exponent
    base, exp = s.split('e')
    # Split mantissa to handle decimal point
    mantissa_parts = base.split('.')
    # Truncate after desired precision
    if len(mantissa_parts) > 1:
        truncated = mantissa_parts[0] + '.' + mantissa_parts[1][:precision]
    else:
        truncated = mantissa_parts[0]
    # Reconstruct scientific notation
    return f"{truncated}e{exp}"

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
    with open(filename, "r") as file:
        for line in file:
            values = line.strip().split(",")
            data.extend([float(val) for val in values])
            count += len(values)
            if count >= total_elements:
                break
    # Ensure we got enough data
    if len(data) < total_elements:
        raise ValueError(
            f"Not enough elements in file. Expected {total_elements}, got {len(data)}"
        )
    # Only keep the exact number we need
    data = np.array(data[:total_elements]).astype(np.float64)
    print(f"Data loaded, size: {data.size}")
    # Reshape into a matrix
    matrix = data.reshape(size, size)
    return matrix

def matrix_multiply_benchmark(file_a, file_b, size=2):
    try:
        print(f"Starting benchmark with files: {file_a} and {file_b}")
        A = read_matrix_from_csv(file_a, size)
        B = read_matrix_from_csv(file_b, size)
        print("Running warm-up multiplication...")
        _ = A @ B
        print("Running timed multiplication...")
        start = time.perf_counter()
        C = A @ B
        end = time.perf_counter()
        operations = 2 * size * size * size
        time_taken = end - start
        gflops = (operations / time_taken) / 1e9
        print(f"Matrix size: {size}x{size}")
        print(f"Time taken: {time_taken:.4f} seconds")
        print(f"Performance: {gflops:.2f} GFLOPS/s")
        print("Saving result...")
        
        # Write result with custom truncation formatting
        with open("data/numpyResult.csv", 'w') as f:
            for row in C:
                # Format each value with truncation instead of rounding
                formatted_row = [truncate_formatter(x, precision=5) for x in row]
                f.write(','.join(formatted_row) + '\n')
                
        print("Result saved to data/numpyResult.csv")
        return C
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    result = matrix_multiply_benchmark("data/A.csv", "data/B.csv", 2048)
