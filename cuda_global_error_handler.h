#ifndef CUDA_GLOBAL_ERROR_HANDLER_H
#define CUDA_GLOBAL_ERROR_HANDLER_H

#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <string>

// Error codes
enum ErrorCode
{
    NO_ERROR          = 0,
    KERNEL_ERROR      = 1,
    MEMORY_ERROR      = 2,
    COMPUTATION_ERROR = 3
};

// Error status for atomic operations
enum AtomicStatus
{
    ATOMIC_NO_ERROR       = 0,
    ATOMIC_ERROR_REPORTED = 1
};

// Error payload structure
struct KernelError
{
    int error_code;
    int line_number;
    int block_id;
    int thread_id;
    char message[256];
    char kernel_name[64];
};

// Device-side global pointers (one copy per translation unit)
static __device__ KernelError * g_device_error_ptr;
static __device__ int * g_device_status_ptr;

// Global error reporting system
class GlobalErrorReporter
{
  private:
    static GlobalErrorReporter * instance;
    static std::mutex init_mutex;

    KernelError * host_error_data;   // Host-accessible pinned memory
    KernelError * device_error_data; // Device pointer to same memory
    int * host_status;               // Host-accessible status flag
    int * device_status;             // Device pointer to status flag
    bool initialized;

    // Private constructor for singleton
    GlobalErrorReporter() : initialized( false )
    {
        initialize();
    }

    void initialize()
    {
        if( initialized )
            return;

        // Check if device supports mapped pinned memory
        int device;
        cudaGetDevice( &device );

        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, device );

        if( !prop.canMapHostMemory )
        {
            std::cerr << "Device does not support mapped pinned memory!" << std::endl;
            return;
        }

        // Enable mapped pinned memory
        cudaSetDeviceFlags( cudaDeviceMapHost );

        // Allocate pinned memory for error data
        cudaError_t err = cudaHostAlloc(
            (void **)&host_error_data, sizeof( KernelError ), cudaHostAllocMapped | cudaHostAllocWriteCombined );
        if( err != cudaSuccess )
        {
            std::cerr << "Failed to allocate pinned memory for error data: " << cudaGetErrorString( err ) << std::endl;
            return;
        }

        err = cudaHostGetDevicePointer( (void **)&device_error_data, host_error_data, 0 );
        if( err != cudaSuccess )
        {
            std::cerr << "Failed to get device pointer for error data: " << cudaGetErrorString( err ) << std::endl;
            cudaFreeHost( host_error_data );
            return;
        }

        // Allocate pinned memory for status flag
        err = cudaHostAlloc( (void **)&host_status, sizeof( int ), cudaHostAllocMapped | cudaHostAllocWriteCombined );
        if( err != cudaSuccess )
        {
            std::cerr << "Failed to allocate pinned memory for status: " << cudaGetErrorString( err ) << std::endl;
            cudaFreeHost( host_error_data );
            return;
        }

        err = cudaHostGetDevicePointer( (void **)&device_status, host_status, 0 );
        if( err != cudaSuccess )
        {
            std::cerr << "Failed to get device pointer for status: " << cudaGetErrorString( err ) << std::endl;
            cudaFreeHost( host_error_data );
            cudaFreeHost( host_status );
            return;
        }

        // Copy device pointers to device-side global variables
        err = cudaMemcpyToSymbol( g_device_error_ptr, &device_error_data, sizeof( KernelError * ) );
        if( err != cudaSuccess )
        {
            std::cerr << "Failed to copy error pointer to device: " << cudaGetErrorString( err ) << std::endl;
            cudaFreeHost( host_error_data );
            cudaFreeHost( host_status );
            return;
        }

        err = cudaMemcpyToSymbol( g_device_status_ptr, &device_status, sizeof( int * ) );
        if( err != cudaSuccess )
        {
            std::cerr << "Failed to copy status pointer to device: " << cudaGetErrorString( err ) << std::endl;
            cudaFreeHost( host_error_data );
            cudaFreeHost( host_status );
            return;
        }

        // Initialize to no error state
        clearError();
        initialized = true;
        std::cout << "Global error reporter has been initialized." << std::endl;
    }

  public:
    // Singleton access
    static GlobalErrorReporter & getInstance()
    {
        std::lock_guard<std::mutex> lock( init_mutex );
        if( !instance )
        {
            instance = new GlobalErrorReporter();
        }
        return *instance;
    }

    // Cleanup (call at program exit)
    static void cleanup()
    {
        std::lock_guard<std::mutex> lock( init_mutex );
        if( instance )
        {
            // Clear device-side global pointers
            KernelError * null_error_ptr = nullptr;
            int * null_status_ptr        = nullptr;
            cudaMemcpyToSymbol( g_device_error_ptr, &null_error_ptr, sizeof( KernelError * ) );
            cudaMemcpyToSymbol( g_device_status_ptr, &null_status_ptr, sizeof( int * ) );

            if( instance->host_error_data )
                cudaFreeHost( instance->host_error_data );
            if( instance->host_status )
                cudaFreeHost( instance->host_status );
            delete instance;
            instance = nullptr;
        }
        std::cout << "Global error reporter has been cleaned up." << std::endl;
    }

    // Get device pointers for kernel use
    KernelError * getDeviceErrorPtr()
    {
        return initialized ? device_error_data : nullptr;
    }
    int * getDeviceStatusPtr()
    {
        return initialized ? device_status : nullptr;
    }

    // Host-side error checking
    bool hasError() const
    {
        return initialized && ( *host_status == ATOMIC_ERROR_REPORTED );
    }

    // Get error data (call only if hasError() returns true)
    const KernelError & getError() const
    {
        return *host_error_data;
    }

    // Clear error status
    void clearError()
    {
        if( initialized )
        {
            *host_status = ATOMIC_NO_ERROR;
            memset( host_error_data, 0, sizeof( KernelError ) );
        }
    }

    bool isInitialized() const
    {
        return initialized;
    }
};

// Static member definitions (put these in a .cpp file in real projects)
GlobalErrorReporter * GlobalErrorReporter::instance = nullptr;
std::mutex GlobalErrorReporter::init_mutex;

// Global functions for easy access from device code
__device__ inline KernelError * getGlobalErrorPtr()
{
    return g_device_error_ptr;
}

__device__ inline int * getGlobalStatusPtr()
{
    return g_device_status_ptr;
}

// Device function to report error (called from kernel)
__device__ void
report_kernel_error( int error_code, const char * message = "", const char * kernel_name = "", int line_number = 0 )
{
    KernelError * error = getGlobalErrorPtr();
    int * status        = getGlobalStatusPtr();

    if( error == nullptr or status == nullptr )
        return; // Not initialized

    // Use atomicCAS to ensure only first error is reported
    if( atomicCAS( status, ATOMIC_NO_ERROR, ATOMIC_ERROR_REPORTED ) == ATOMIC_NO_ERROR )
    {
        // This thread gets to report the error
        error->error_code  = error_code;
        error->line_number = line_number;
        error->block_id    = blockIdx.x;
        error->thread_id   = threadIdx.x;

        // Copy message
        int i = 0;
        while( message[i] != '\0' && i < 255 )
        {
            error->message[i] = message[i];
            i++;
        }
        error->message[i] = '\0';

        // Copy kernel name
        i = 0;
        while( kernel_name[i] != '\0' && i < 63 )
        {
            error->kernel_name[i] = kernel_name[i];
            i++;
        }
        error->kernel_name[i] = '\0';

        // Ensure all writes are visible to host
        __threadfence_system();
    }
}

// Macro for convenient error reporting with automatic line number and kernel name
#define REPORT_ERROR( code, msg )                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        report_kernel_error( code, msg, __FUNCTION__, __LINE__ );                                                      \
    } while( 0 )

class CudaException : public std::runtime_error
{
  public:
    CudaException( const std::string & message ) : std::runtime_error( message ) {}
};

// Host-side function to check and print kernel errors (similar to your macro requirement)
inline void checkKernelError(
    const char * kernel_name = "unknown", const char * file_name = "unknown", const char * function_name = "unknown",
    int line_number = 0 )
{
    // Check standard CUDA errors first
    cudaError_t cuda_err = cudaGetLastError();
    if( cuda_err != cudaSuccess )
    {
        std::string error_message =
            "CUDA Error in " + std::string( kernel_name ) + ": " + cudaGetErrorString( cuda_err );
        error_message += "\nDetected in function " + std::string( function_name ) + " in " + std::string( file_name ) +
                         ":" + std::to_string( line_number );
        throw CudaException( error_message );
    }

    // Synchronize to ensure kernel completion
    cuda_err = cudaDeviceSynchronize();
    if( cuda_err != cudaSuccess )
    {
        std::string error_message =
            "CUDA Synchronization Error after " + std::string( kernel_name ) + ": " + cudaGetErrorString( cuda_err );
        error_message += "\nDetected in function " + std::string( function_name ) + " in " + std::string( file_name ) +
                         ":" + std::to_string( line_number );
        throw CudaException( error_message );
    }

    // Check for kernel-reported errors
    GlobalErrorReporter & reporter = GlobalErrorReporter::getInstance();
    if( reporter.hasError() )
    {
        const auto & error        = reporter.getError();
        std::string error_message = "Kernel Error in " + std::string( error.kernel_name ) + " at line " +
                                    std::to_string( error.line_number ) + ": " + error.message;
        error_message += "\nDetected in function " + std::string( function_name ) + " in " + std::string( file_name ) +
                         ":" + std::to_string( line_number );
        reporter.clearError();
        throw CudaException( error_message );
    }
}

// Macro for checking errors after kernel launch (your centrally defined macro)
#define CHECK_KERNEL()                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        checkKernelError( "unknown kernel", __FILE__, __FUNCTION__, __LINE__ );                                        \
    } while( 0 )

// Alternative macro that takes kernel name
#define CHECK_KERNEL_NAMED( name )                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        checkKernelError( name, __FILE__, __FUNCTION__, __LINE__ );                                                    \
    } while( 0 )

// RAII helper for automatic cleanup
class CudaErrorCleanup
{
  public:
    ~CudaErrorCleanup()
    {
        GlobalErrorReporter::cleanup();
    }
};

// Global cleanup object (include this in one .cpp file)
static CudaErrorCleanup g_cuda_cleanup;

#endif // CUDA_GLOBAL_ERROR_HANDLER_H
