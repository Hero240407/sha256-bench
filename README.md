# OpenCL and CPU SHA256 Hashing Benchmark

This project provides a Python script to benchmark SHA256 hashing performance on a system's GPU (using OpenCL) and CPU. It measures the number of hashes generated per second (H/s) and provides statistics like average, minimum, and maximum H/s over multiple iterations.

## Features

*   **GPU Benchmarking:** Utilizes OpenCL to run SHA256 hashing on a selected GPU.
*   **CPU Benchmarking:** Uses Python's built-in `hashlib` and `multiprocessing` for parallel SHA256 hashing on all available CPU cores.
*   **Warm-up Phase:** Includes a configurable warm-up period for both GPU and CPU before each benchmark iteration to help stabilize hardware performance (clocks, thermals) for more consistent results.
*   **Interleaved Benchmarking:** For each iteration, the script follows this sequence: Warm up GPU -> Benchmark GPU -> Warm up CPU -> Benchmark CPU.
*   **Configurable Parameters:** Allows easy adjustment of the number of iterations, hashes per run, local work size for GPU, and warm-up duration.
*   **Device Selection:** Prompts the user to select an OpenCL device if multiple are available, with an attempt to auto-select a suitable GPU.
*   **Performance Metrics:** Calculates and displays average, minimum, and maximum Hashes/second (H/s) for both CPU and GPU.

## How it Works

1.  **GPU Hashing (`sha256_kernel.cl` & PyOpenCL):**
    *   An OpenCL kernel (`sha256_kernel.cl`) implements a basic SHA256 algorithm.
    *   Each work-item in the OpenCL kernel processes a unique 4-byte nonce to generate a SHA256 hash.
    *   The Python script (`benchmark.py`) uses `PyOpenCL` to compile this kernel, manage memory buffers on the GPU, and dispatch a large number of hash computations.
    *   The script attempts to maximize GPU utilization by launching a substantial number of hashes per run, configured by `TARGET_HASHES_GPU` and `LWS_GPU_CONFIG`.

2.  **CPU Hashing (`hashlib` & `multiprocessing`):**
    *   The Python script uses the `hashlib.sha256()` function for hashing on the CPU.
    *   To utilize all CPU cores and maximize CPU load, the `multiprocessing` module is used to distribute the hashing tasks across multiple worker processes.
    *   Each CPU worker hashes a portion of the total `HASHES_PER_RUN_CPU`.

3.  **Warm-up & Benchmarking Sequence:**
    *   The script first initializes OpenCL and selects a device.
    *   For each of the `NUM_ITERATIONS`:
        1.  **GPU Warm-up:** The GPU is loaded with hashing tasks for `WARMUP_DURATION_SECONDS`.
        2.  **GPU Benchmark:** A timed benchmark run computes `HASHES_PER_RUN_GPU` hashes on the GPU, and H/s is recorded.
        3.  **CPU Warm-up:** All CPU cores are loaded with hashing tasks for `WARMUP_DURATION_SECONDS`.
        4.  **CPU Benchmark:** A timed benchmark run computes `HASHES_PER_RUN_CPU` hashes on the CPU, and H/s is recorded.
    *   After all iterations, summary statistics (average, min, max H/s) are displayed.

## Files

*   `benchmark.py`: The main Python script that orchestrates the benchmark.
*   `sha256_kernel.cl`: The OpenCL C kernel code for SHA256 hashing on the GPU.
*   `README.md`: This file.

## Dependencies & Setup

1.  **Python:** Python 3.7 or newer is recommended.
2.  **Python Libraries:** Install the necessary libraries using pip:
    ```bash
    pip install pyopencl numpy
    ```
3.  **OpenCL SDK/Driver:**
    *   You **must** have an OpenCL SDK (Software Development Kit) or a compatible driver installed for your GPU.
        *   **NVIDIA GPUs:** Install the NVIDIA CUDA Toolkit (which includes an OpenCL driver).
        *   **AMD GPUs:** Install AMD ROCm (for Linux) or the latest AMD Radeon Software / AMD Software: Adrenalin Edition (which includes the OpenCL driver) for Windows or Linux. Older systems might use the AMD APP SDK.
        *   **Intel GPUs/CPUs:** Install the Intel OpenCL SDK or relevant drivers (e.g., Intel Graphics Compute Runtime for OpenCL).
    *   Ensure your GPU drivers are up-to-date.
4.  **Verification (Optional but Recommended):**
    *   You can check if OpenCL is correctly installed and devices are visible using tools like `clinfo` (common on Linux) or by running a simple PyOpenCL device query script.

## How to Run

1.  Ensure both `benchmark.py` and `sha256_kernel.cl` are in the same directory.
2.  Open a terminal or command prompt in that directory.
3.  Execute the script:
    ```bash
    python benchmark.py
    ```
4.  If multiple OpenCL devices are found, the script will list them and attempt to auto-select a GPU. You might be prompted to choose if auto-selection is not straightforward.
5.  The script will then proceed with the warm-up and benchmarking iterations.

## Script Configuration

The following parameters can be adjusted at the top of `benchmark.py`:

*   `NUM_ITERATIONS`: Number of benchmark cycles to run for averaging (e.g., `3`).
*   `LWS_GPU_CONFIG`: A tuple defining the local work size for the GPU kernel (e.g., `(256,)`). This is the number of work-items in a work-group. `256` is often a good starting point for AMD GPUs.
*   `TARGET_HASHES_GPU`: The approximate target number of hashes to compute in a single GPU benchmark run (e.g., `1024 * 1024 * 50` for ~52 million). The actual `HASHES_PER_RUN_GPU` will be adjusted to be a multiple of `LWS_GPU_CONFIG[0]`.
*   `HASHES_PER_RUN_CPU`: The number of hashes for a single CPU benchmark run (e.g., `1024 * 100`).
*   `WARMUP_DURATION_SECONDS`: Duration in seconds for the warm-up phase before each benchmark (e.g., `5`).
*   `HASHES_PER_GPU_WARMUP_BATCH`: Number of hashes computed in each sub-batch during GPU warmup. Derived from `HASHES_PER_RUN_GPU` and `LWS_GPU_CONFIG`.

## Python Script Functions (`benchmark.py`)

*   **`get_opencl_device()`**:
    *   Queries available OpenCL platforms and devices.
    *   Attempts to auto-select a suitable GPU based on `LWS_GPU_CONFIG`.
    *   Prompts for manual selection if needed.
    *   Checks if the selected GPU's `max_work_group_size` is compatible with `LWS_GPU_CONFIG` and adjusts `effective_lws_gpu` if necessary.
*   **`benchmark_gpu_internal(...)`**:
    *   The core logic for executing OpenCL kernels for a given number of hashes.
    *   Manages OpenCL buffer creation, kernel argument setting, and kernel execution.
    *   Handles data copy back from GPU to host (if not in warmup mode).
    *   Used by both GPU warmup and main GPU benchmark.
*   **`cpu_hash_worker(...)`**:
    *   The target function for each CPU worker process created by `multiprocessing`.
    *   Performs a specified number of SHA256 hashes using `hashlib`.
*   **`warmup_device(...)`**:
    *   Runs hashing computations on the specified device type ("GPU" or "CPU") for `WARMUP_DURATION_SECONDS`.
    *   For GPU, it repeatedly calls `benchmark_gpu_internal` with `is_warmup=True`.
    *   For CPU, it uses a `multiprocessing.Pool` to load all CPU cores.
*   **`benchmark_gpu_main(...)`**:
    *   Orchestrates a single, timed GPU benchmark iteration.
    *   Calls `benchmark_gpu_internal` and calculates H/s.
*   **`benchmark_cpu_main(...)`**:
    *   Orchestrates a single, timed CPU benchmark iteration using `multiprocessing.Pool` and `cpu_hash_worker`.
    *   Calculates H/s.
*   **`if __name__ == "__main__":` block**:
    *   Main execution entry point.
    *   Initializes script configurations (adjusting hash counts to be multiples of LWS).
    *   Initializes OpenCL context, queue, and compiles the GPU program once.
    *   Contains the main loop that iterates `NUM_ITERATIONS` times, calling the warm-up and benchmark functions for GPU and CPU in an interleaved manner.
    *   Collects results and prints the final summary.

## OpenCL Kernel (`sha256_kernel.cl`)

*   **Purpose:** Implements a basic SHA256 hashing algorithm designed to run on an OpenCL-compliant device (typically a GPU).
*   **Key Components:**
    *   **Preprocessor Macros:** Defines SHA256 helper functions like `ROTR`, `SHR`, `Ch`, `Maj`, `Sigma0`, `Sigma1`, `sigma0_small`, `sigma1_small`.
    *   **`k[]` Array:** Stores the SHA256 round constants. Declared as `__constant static const` for optimal GPU memory usage.
    *   **`sha256_transform_nonce(uint nonce_val, __global uint* hash_out)`:**
        *   The core function that takes a single `uint` nonce.
        *   Pads the nonce to form a 512-bit message block (as per SHA256 standard for a 4-byte message).
        *   Performs the 64 rounds of SHA256 computation.
        *   Writes the resulting 32-byte (8 `uint`s) hash to the global output buffer `hash_out`.
    *   **`__kernel void hash_main(__global uint* output_hashes, uint start_nonce, uint num_hashes_per_kernel_call)`:**
        *   The main OpenCL kernel function launched from Python.
        *   `get_global_id(0)` gives each work-item a unique ID.
        *   Calculates a `current_nonce` based on `start_nonce` and the global ID.
        *   Calls `sha256_transform_nonce` to compute the hash for `current_nonce`.
        *   The output hash is stored at the correct offset in `output_hashes`.

## Troubleshooting & Notes

*   **OpenCL Build Errors:** If the script reports "OpenCL Kernel Compilation Failed!", carefully examine the "Build Log" printed in the console. It will contain specific error messages from the OpenCL compiler for your GPU, indicating syntax errors or other issues in `sha256_kernel.cl` or incompatibilities with your OpenCL version/driver.
*   **Low GPU Utilization:** If GPU utilization is low and spiky (monitor with tools like Task Manager on Windows, `radeontop`/`nvtop` on Linux), it usually means `HASHES_PER_RUN_GPU` is too small for your GPU. The Python overhead of launching kernels dominates. Increase `TARGET_HASHES_GPU`. The current values are tuned for better utilization.
*   **VRAM Limitations:** If `TARGET_HASHES_GPU` is set extremely high, the script might fail when trying to allocate the output buffer on the GPU due to insufficient VRAM. The script will print the buffer size in MB if this occurs. Reduce `TARGET_HASHES_GPU` if this happens.
*   **Nonce Overflow:** Nonce values for the GPU are handled as `uint32` and are wrapped using modulo arithmetic (`% UINT32_MAX`) to prevent Python integer overflow issues before conversion to `np.uint32`.
*   **`effective_lws_gpu`:** The script attempts to use the `LWS_GPU_CONFIG`. If the selected device's `max_work_group_size` is smaller, it tries to find a suitable power-of-2 LWS or defaults to `None` (letting OpenCL decide), which might impact performance. Relevant messages will be printed.
