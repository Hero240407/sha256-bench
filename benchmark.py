import pyopencl as cl
import numpy as np
import time
import hashlib
import multiprocessing
import os

NUM_ITERATIONS = 3
LWS_GPU_CONFIG = (256,) 

TARGET_HASHES_GPU = 1024 * 1024 * 50 
HASHES_PER_RUN_GPU = (TARGET_HASHES_GPU // LWS_GPU_CONFIG[0]) * LWS_GPU_CONFIG[0]

HASHES_PER_RUN_CPU = 1024 * 100 

WARMUP_DURATION_SECONDS = 2
UINT32_MAX = 2**32

TARGET_GPU_WARMUP_BATCH = HASHES_PER_RUN_GPU // 10
HASHES_PER_GPU_WARMUP_BATCH = (TARGET_GPU_WARMUP_BATCH // LWS_GPU_CONFIG[0]) * LWS_GPU_CONFIG[0]
if HASHES_PER_GPU_WARMUP_BATCH == 0: 
    HASHES_PER_GPU_WARMUP_BATCH = LWS_GPU_CONFIG[0]


SHA256_KERNEL_SRC = ""
try:
    with open("sha256_kernel.cl", "r") as f:
        SHA256_KERNEL_SRC = f.read()
except FileNotFoundError:
    print("CRITICAL ERROR: sha256_kernel.cl not found. Please create it in the same directory.")
    exit()
except Exception as e:
    print(f"CRITICAL ERROR: Could not read sha256_kernel.cl: {e}")
    exit()

if not SHA256_KERNEL_SRC.strip():
    print("CRITICAL ERROR: sha256_kernel.cl is empty.")
    exit()

def get_opencl_device():
    platforms = cl.get_platforms()
    if not platforms:
        print("No OpenCL platforms found! Make sure OpenCL drivers/SDK are installed.")
        return None, None

    print("\nAvailable OpenCL Platforms and Devices:")
    all_devices = []
    for i, platform in enumerate(platforms):
        print(f"Platform {i}: {platform.name} (Version: {platform.version})")
        try:
            devices = platform.get_devices()
            if not devices:
                print(f"  No devices found for platform {platform.name}.")
                continue
            for j, device in enumerate(devices):
                max_wg_size = device.max_work_group_size
                print(f"  Device {j}: {device.name} (Type: {cl.device_type.to_string(device.type)}, Version: {device.version}, Max WG Size: {max_wg_size})")
                all_devices.append((platform, device))
        except cl.RuntimeError as e:
            print(f"  Error getting devices for platform {platform.name}: {e}")

    if not all_devices:
        print("No OpenCL devices found on any platform.")
        return None, None

    gpu_devices = [(p, d) for p, d in all_devices if d.type == cl.device_type.GPU]
    if gpu_devices:
        suitable_gpus = [(p, d) for p, d in gpu_devices if d.max_work_group_size >= LWS_GPU_CONFIG[0]]
        if suitable_gpus:
            print(f"\nAuto-selecting first suitable GPU: {suitable_gpus[0][1].name}")
            return suitable_gpus[0]
        elif gpu_devices:
             print(f"\nWarning: No GPU found supporting local work size {LWS_GPU_CONFIG[0]}. Auto-selecting first GPU: {gpu_devices[0][1].name}. This might fail or perform poorly.")
             return gpu_devices[0]

    cpu_devices = [(p, d) for p, d in all_devices if d.type == cl.device_type.CPU]
    if cpu_devices:
        print(f"\nNo suitable GPU found. Auto-selecting first CPU OpenCL device: {cpu_devices[0][1].name}")
        return cpu_devices[0]

    print("\nCould not auto-select a GPU or CPU OpenCL device. Please choose a device manually.")
    try:
        flat_device_list = [(p,d) for p,d in all_devices]
        for idx, (p,d) in enumerate(flat_device_list):
            print(f"  {idx}: {d.name} (on Platform: {p.name})")
        choice = int(input(f"Select Device index (0-{len(flat_device_list)-1}): "))
        if 0 <= choice < len(flat_device_list):
            selected_p_d = flat_device_list[choice]
            if selected_p_d[1].type == cl.device_type.GPU and selected_p_d[1].max_work_group_size < LWS_GPU_CONFIG[0]:
                print(f"Warning: Selected GPU {selected_p_d[1].name} has max work group size {selected_p_d[1].max_work_group_size}, less than configured {LWS_GPU_CONFIG[0]}.")
            return selected_p_d
        else:
            print("Invalid selection. Exiting.")
            return None, None
    except (ValueError, IndexError) as e:
        print(f"Invalid selection: {e}. Exiting.")
        return None, None

def benchmark_gpu_internal(ctx, queue, prg, num_hashes_total, local_work_size_config, current_run_nonce_offset=0, is_warmup=False):
    output_buffer_size = num_hashes_total * 8 * np.dtype(np.uint32).itemsize
    mf = cl.mem_flags
    try:
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY | (mf.ALLOC_HOST_PTR if not is_warmup else mf.WRITE_ONLY), size=output_buffer_size)
        if not is_warmup:
             output_hashes_np = np.empty(num_hashes_total * 8, dtype=np.uint32)
        else:
             output_hashes_np = None
    except cl.Error as e:
        if not is_warmup:
            print(f"Error creating OpenCL output buffer (size: {output_buffer_size/1024/1024:.2f} MB): {e}")
            if "CL_MEM_OBJECT_ALLOCATION_FAILURE" in str(e) or "out of memory" in str(e).lower():
                print("This is likely due to insufficient VRAM for the requested number of hashes.")
                print(f"Try reducing HASHES_PER_RUN_GPU (currently attempting {num_hashes_total:,} hashes).")
        return 0

    start_nonce_val = current_run_nonce_offset % UINT32_MAX
    start_nonce = np.uint32(start_nonce_val) 
    num_hashes_for_kernel_call = np.uint32(num_hashes_total)
    
    global_work_size = (num_hashes_total,)
    local_work_size = local_work_size_config

    try:
        kernel_event = prg.hash_main(queue, global_work_size, local_work_size,
                                     output_buf, start_nonce, num_hashes_for_kernel_call)
        if not is_warmup:
            copy_event = cl.enqueue_copy(queue, output_hashes_np, output_buf, wait_for=[kernel_event])
            copy_event.wait()
        else:
            kernel_event.wait()
            
    except cl.Error as e:
        if not is_warmup:
            print(f"Error during OpenCL kernel execution or data copy with GWS={global_work_size}, LWS={local_work_size}: {e}")
        return 0
    return num_hashes_total


def cpu_hash_worker(start_nonce, num_hashes, base_data_str=""):
    count = 0
    for i in range(num_hashes):
        nonce = start_nonce + i 
        data_to_hash = (f"{base_data_str}{nonce:08d}").encode('utf-8')
        hashlib.sha256(data_to_hash).digest()
        count += 1
    return count

def warmup_device(duration_seconds, device_type, local_work_size_gpu_cfg=None, platform=None, device=None, ctx=None, queue=None, prg=None):
    print(f"\n--- Warming up {device_type} for {duration_seconds} seconds... ---")
    start_warmup_time = time.perf_counter()
    total_hashes_warmed_up = 0
    
    if device_type == "GPU":
        if not all([platform, device, ctx, queue, prg, local_work_size_gpu_cfg]):
            print("GPU Warmup: Missing OpenCL components or local work size config. Skipping.")
            return
        
        if HASHES_PER_GPU_WARMUP_BATCH == 0:
            print("GPU Warmup: Batch size is zero. Skipping.")
            return

        gpu_warmup_nonce_offset = int(time.time() * 1000) % UINT32_MAX

        while time.perf_counter() - start_warmup_time < duration_seconds:
            benchmark_gpu_internal(ctx, queue, prg, HASHES_PER_GPU_WARMUP_BATCH, 
                                   local_work_size_gpu_cfg, gpu_warmup_nonce_offset, is_warmup=True)
            gpu_warmup_nonce_offset = (gpu_warmup_nonce_offset + HASHES_PER_GPU_WARMUP_BATCH) % UINT32_MAX
            total_hashes_warmed_up += HASHES_PER_GPU_WARMUP_BATCH
            if time.perf_counter() - start_warmup_time >= duration_seconds:
                break
    
    elif device_type == "CPU":
        num_workers = os.cpu_count() or 1
        CPU_WARMUP_BATCH_PER_WORKER = max(1024, HASHES_PER_RUN_CPU // (num_workers * 10) ) 
        if CPU_WARMUP_BATCH_PER_WORKER == 0: CPU_WARMUP_BATCH_PER_WORKER = 1024
        
        cpu_warmup_nonce_offset = 0

        with multiprocessing.Pool(processes=num_workers) as pool:
            while time.perf_counter() - start_warmup_time < duration_seconds:
                tasks = []
                current_batch_start_nonce = cpu_warmup_nonce_offset
                for i in range(num_workers):
                    tasks.append((current_batch_start_nonce, CPU_WARMUP_BATCH_PER_WORKER, f"warmup_worker_{i}_"))
                    current_batch_start_nonce += CPU_WARMUP_BATCH_PER_WORKER
                
                pool.starmap(cpu_hash_worker, tasks)
                cpu_warmup_nonce_offset += (CPU_WARMUP_BATCH_PER_WORKER * num_workers)
                total_hashes_warmed_up += (CPU_WARMUP_BATCH_PER_WORKER * num_workers)
                if time.perf_counter() - start_warmup_time >= duration_seconds:
                    break
    
    elapsed_warmup_time = time.perf_counter() - start_warmup_time
    print(f"--- {device_type} warmup complete. Duration: {elapsed_warmup_time:.2f}s. Hashes computed (approx): {total_hashes_warmed_up:,} ---")

def benchmark_gpu_main(platform, device, num_hashes_total, local_work_size_config, current_run_nonce_offset=0, ctx=None, queue=None, prg=None):
    if device is None or not all([ctx, queue, prg, local_work_size_config]):
        print("GPU Benchmark: Device, OpenCL components, or LWS config not available.")
        return 0
        
    print(f"\n--- GPU Benchmark on: {device.name} (LWS: {local_work_size_config[0] if local_work_size_config else 'Auto'}) ---")
    print(f"Attempting to compute {num_hashes_total:,} hashes...")

    start_time = time.perf_counter()
    hashes_computed = benchmark_gpu_internal(ctx, queue, prg, num_hashes_total, 
                                             local_work_size_config, current_run_nonce_offset, is_warmup=False)
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    if hashes_computed > 0 and duration > 0:
        hashes_per_second = hashes_computed / duration
        print(f"GPU: Computed {hashes_computed:,} hashes in {duration:.4f}s")
        print(f"GPU H/s: {hashes_per_second:,.2f}")
        return hashes_per_second
    else:
        print("GPU: Benchmark run failed, duration was zero, or no hashes computed.")
        return 0

def benchmark_cpu_main(num_hashes_total, base_data_str=""):
    print(f"\n--- CPU Benchmark ---")
    num_workers = os.cpu_count() or 1
    print(f"Using {num_workers} CPU workers.")
    print(f"Attempting to compute {num_hashes_total:,} hashes...")

    if num_hashes_total == 0:
        print("CPU: No hashes to compute.")
        return 0

    hashes_per_worker = num_hashes_total // num_workers
    remaining_hashes = num_hashes_total % num_workers
    tasks = []
    current_task_start_nonce = 0
    for i in range(num_workers):
        count_for_this_worker = hashes_per_worker + (1 if i < remaining_hashes else 0)
        if count_for_this_worker > 0:
            tasks.append((current_task_start_nonce, count_for_this_worker, base_data_str + f"worker_{i}_"))
            current_task_start_nonce += count_for_this_worker
    
    start_time = time.perf_counter()
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(cpu_hash_worker, tasks)
    except Exception as e:
        print(f"Error during CPU multiprocessing: {e}")
        return 0
    total_hashes_done = sum(results)
    end_time = time.perf_counter()
    duration = end_time - start_time

    if duration > 0:
        hashes_per_second = total_hashes_done / duration
        print(f"CPU: Computed {total_hashes_done:,} hashes in {duration:.4f} s")
        print(f"CPU H/s: {hashes_per_second:,.2f}")
        return hashes_per_second
    else:
        print("CPU: Duration was too short to measure or an error occurred.")
        return 0

if __name__ == "__main__":
    print(f"Configured HASHES_PER_RUN_GPU: {HASHES_PER_RUN_GPU:,}")
    print(f"Configured HASHES_PER_GPU_WARMUP_BATCH: {HASHES_PER_GPU_WARMUP_BATCH:,}")
    print(f"Configured LWS_GPU_CONFIG: {LWS_GPU_CONFIG[0]}")

    gpu_hps_results = []
    cpu_hps_results = []

    print("\nInitializing OpenCL...")
    selected_platform, selected_device = get_opencl_device()

    gpu_ctx, gpu_queue, gpu_prg = None, None, None
    effective_lws_gpu = LWS_GPU_CONFIG 

    if selected_device and selected_device.type == cl.device_type.GPU:
        if selected_device.max_work_group_size < LWS_GPU_CONFIG[0]:
            print(f"Warning: Device {selected_device.name} max work group size ({selected_device.max_work_group_size}) is less than configured LWS ({LWS_GPU_CONFIG[0]}).")
            temp_lws_val = selected_device.max_work_group_size
            while temp_lws_val & (temp_lws_val -1 ) !=0 and temp_lws_val > 0 : 
                temp_lws_val -=1
            if temp_lws_val >= 64:
                effective_lws_gpu = (temp_lws_val,)
                print(f"Adjusting LWS to device max compatible power of 2: {effective_lws_gpu[0]}")
                HASHES_PER_RUN_GPU = (TARGET_HASHES_GPU // effective_lws_gpu[0]) * effective_lws_gpu[0]
                TARGET_GPU_WARMUP_BATCH = HASHES_PER_RUN_GPU // 10
                HASHES_PER_GPU_WARMUP_BATCH = (TARGET_GPU_WARMUP_BATCH // effective_lws_gpu[0]) * effective_lws_gpu[0]
                if HASHES_PER_GPU_WARMUP_BATCH == 0: HASHES_PER_GPU_WARMUP_BATCH = effective_lws_gpu[0]
                print(f"Recalculated HASHES_PER_RUN_GPU: {HASHES_PER_RUN_GPU:,}")
                print(f"Recalculated HASHES_PER_GPU_WARMUP_BATCH: {HASHES_PER_GPU_WARMUP_BATCH:,}")
            else:
                effective_lws_gpu = None 
                print("Cannot find suitable LWS based on device max; letting OpenCL decide (None).")
                HASHES_PER_RUN_GPU = TARGET_HASHES_GPU
                HASHES_PER_GPU_WARMUP_BATCH = max(LWS_GPU_CONFIG[0]*10, HASHES_PER_RUN_GPU // 10)
                if HASHES_PER_GPU_WARMUP_BATCH == 0: HASHES_PER_GPU_WARMUP_BATCH = LWS_GPU_CONFIG[0]*10
        try:
            gpu_ctx = cl.Context([selected_device])
            gpu_queue = cl.CommandQueue(gpu_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
            gpu_prg = cl.Program(gpu_ctx, SHA256_KERNEL_SRC).build(options=[])
            print("OpenCL program for GPU built successfully.")
        except cl.RuntimeError as e: 
            print("\nCRITICAL: OpenCL Kernel Compilation Failed during initial setup!")
            print(f"Error message: {e.what if hasattr(e, 'what') else e}")
            logs = getattr(e, 'build_logs', None)
            if logs:
                for dev, log_content in logs:
                    print(f"\n--- Build Log for device {dev.name} ---\n{log_content}\n--- End Build Log ---")
            else: 
                log_attr = getattr(e, 'log', None)
                if isinstance(log_attr, list) and len(log_attr) > 0 and isinstance(log_attr[0], tuple) and len(log_attr[0]) > 1:
                    print(f"\n--- Build Log ---\n{log_attr[0][1]}\n--- End Build Log ---")
                elif isinstance(log_attr, str):
                     print(f"\n--- Build Log ---\n{log_attr}\n--- End Build Log ---")
                else:
                    print("Could not retrieve detailed build log.")
            selected_device = None 
            gpu_ctx, gpu_queue, gpu_prg = None, None, None 
            print("GPU benchmarking will be skipped due to build failure.")
        except Exception as e:
            print(f"Unexpected error during OpenCL setup: {e}")
            selected_device = None
            gpu_ctx, gpu_queue, gpu_prg = None, None, None
    
    if selected_device and not gpu_prg:
        selected_device = None

    run_base_nonce_offset = (int(time.time()) * 1000) % UINT32_MAX

    for i in range(NUM_ITERATIONS):
        print(f"\n{'='*20} Iteration {i+1}/{NUM_ITERATIONS} {'='*20}")
        if selected_device and gpu_prg:
            warmup_device(WARMUP_DURATION_SECONDS, "GPU", 
                          local_work_size_gpu_cfg=effective_lws_gpu,
                          platform=selected_platform, device=selected_device, 
                          ctx=gpu_ctx, queue=gpu_queue, prg=gpu_prg)
            
            iteration_gpu_nonce_offset = (run_base_nonce_offset + (i * HASHES_PER_RUN_GPU)) % UINT32_MAX
            gpu_hps = benchmark_gpu_main(selected_platform, selected_device, HASHES_PER_RUN_GPU, 
                                         effective_lws_gpu, 
                                         iteration_gpu_nonce_offset, ctx=gpu_ctx, queue=gpu_queue, prg=gpu_prg)
            if gpu_hps > 0:
                gpu_hps_results.append(gpu_hps)
            else:
                print("GPU benchmark for this iteration failed or yielded no result.")
        else:
            if i == 0:
                print("Skipping GPU benchmarks as device/program setup failed or no device selected.")

        warmup_device(WARMUP_DURATION_SECONDS, "CPU")
        
        iteration_cpu_base_data = f"run_{run_base_nonce_offset % 1000000}_iter_{i}_"
        cpu_hps = benchmark_cpu_main(HASHES_PER_RUN_CPU, iteration_cpu_base_data)
        if cpu_hps > 0:
            cpu_hps_results.append(cpu_hps)
        else:
            print("CPU benchmark for this iteration failed or yielded no result.")
        
        if i < NUM_ITERATIONS - 1:
            print("Pausing briefly before next iteration's warmups...")
            time.sleep(1)

    print(f"\n\n{'='*20} Benchmark Summary {'='*20}")
    if gpu_hps_results:
        avg_gpu_hps = sum(gpu_hps_results) / len(gpu_hps_results)
        min_gpu_hps = min(gpu_hps_results)
        max_gpu_hps = max(gpu_hps_results)
        print("\nGPU Results:")
        print(f"  LWS Used: {effective_lws_gpu[0] if effective_lws_gpu else 'Auto'}")
        print(f"  Hashes per run: {HASHES_PER_RUN_GPU:,}")
        print(f"  Successful runs: {len(gpu_hps_results)}/{NUM_ITERATIONS if selected_device and gpu_prg else 0}")
        print(f"  Average H/s: {avg_gpu_hps:,.2f}")
        print(f"  Min H/s:     {min_gpu_hps:,.2f}")
        print(f"  Max H/s:     {max_gpu_hps:,.2f}")
    else:
        print("\nGPU Results: No successful GPU benchmark runs.")

    if cpu_hps_results:
        avg_cpu_hps = sum(cpu_hps_results) / len(cpu_hps_results)
        min_cpu_hps = min(cpu_hps_results)
        max_cpu_hps = max(cpu_hps_results)
        print("\nCPU Results:")
        print(f"  Hashes per run: {HASHES_PER_RUN_CPU:,}")
        print(f"  Successful runs: {len(cpu_hps_results)}/{NUM_ITERATIONS}")
        print(f"  Average H/s: {avg_cpu_hps:,.2f}")
        print(f"  Min H/s:     {min_cpu_hps:,.2f}")
        print(f"  Max H/s:     {max_cpu_hps:,.2f}")
    else:
        print("\nCPU Results: No successful CPU benchmark runs.")

    if gpu_hps_results and cpu_hps_results and avg_cpu_hps > 0:
        print(f"\nGPU is approximately {avg_gpu_hps/avg_cpu_hps:.2f}x faster than CPU on average for this workload.")
    elif gpu_hps_results and not cpu_hps_results:
        print("\nCPU benchmark did not yield results for comparison.")
    elif not gpu_hps_results and cpu_hps_results:
        print("\nGPU benchmark did not yield results for comparison.")
    
    print("\nBenchmark finished.")