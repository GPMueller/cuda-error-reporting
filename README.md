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
cancel operations across multiple streams without corrupting the CUDA context.

**Key capabilities:**
- Global stop token shared across all streams
- Soft trap function that signals errors without context corruption
- Per-stream error detection
- Context remains valid after error for recovery operations
- Uses PTX `exit;` instruction for clean thread termination

**How it works:**
1. A kernel detects an error condition and calls `softTrap()`
2. `softTrap()` atomically sets a global stop token
3. Other kernels can check the stop token via `check_ok()` kernel
4. Errors are recorded per-stream without hardware traps
5. Host can detect errors and recover without restarting the CUDA context


## Running Examples

Requires CUDA Toolkit with nvcc compiler in the `PATH`.

```bash
make all           # Build all examples
make run-all       # Run all examples
make run-soft-trap # Run just the soft trap example
```
