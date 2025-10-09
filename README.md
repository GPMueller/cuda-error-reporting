# cuda-error-reporting

Experimentation in CUDA error reporting and handling.

## Features

### 1. Global Error Handler (`cuda_global_error_handler.h`)
A comprehensive error reporting system that allows kernels to report errors back
to the host without requiring explicit parameter passing.

**Key capabilities:**
- Kernel-side error reporting with automatic context capture
- Thread-safe error reporting using atomics
- Host-side error checking and retrieval
- Macro-based error reporting for ease of use

### 2. Soft Trap Mechanism (`cuda_soft_trap.h`)
A soft error mechanism that allows kernels to signal errors and gracefully
cancel operations across multiple streams with proper error reporting to the host.

**Key capabilities:**
- Global stop token shared across all streams
- Soft trap triggers per-stream errors
- `cudaStreamSynchronize()` returns error codes (`cudaErrorIllegalAddress`)
- Requires `cudaDeviceReset()` for recovery (process continues, no restart needed)

**How it works:**
1. A kernel detects an error condition and calls `softTrap()`
2. `softTrap()` atomically sets a global stop token and exits cleanly
3. After each kernel, launch `check_ok()` to check the stop token
4. `check_ok()` dereferences nullptr when stop token is set, triggering stream error
5. All streams with `check_ok()` report errors via `cudaStreamSynchronize()`
6. Host detects errors and calls `cudaDeviceReset()` to recover


## Running Examples

Requires CUDA Toolkit with nvcc compiler in the `PATH`.

```bash
make all           # Build all examples
make run-all       # Run all examples
make run-soft-trap # Run just the soft trap example
```
