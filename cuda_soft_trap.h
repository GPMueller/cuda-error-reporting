#ifndef CUDA_SOFT_TRAP_H
#define CUDA_SOFT_TRAP_H

#include <cuda_runtime.h>
#include <iostream>

// Maximum number of streams we can track
#define MAX_STREAMS 16

// Device-side global variables
static __device__ int g_stop_token                      = 0;
static __device__ int g_stream_error_flags[MAX_STREAMS] = { 0 };

// Device function to trigger soft trap
// This sets the global stop token and exits the current thread cleanly
__device__ void softTrap()
{
    // Atomically set stop token to 1
    atomicExch( &g_stop_token, 1 );

    // Ensure the write is visible across the entire system
    __threadfence_system();

    // Exit the current thread using PTX
    // This terminates the thread cleanly without triggering any errors
    asm volatile( "exit;" );
}

// Kernel to check if stop token is set
// If stop token is set, this kernel records an error and triggers a stream error via nullptr dereference
__global__ void check_ok_kernel( int stream_id )
{
    // Only thread 0 of block 0 checks and records
    if( blockIdx.x == 0 && threadIdx.x == 0 )
    {
        if( g_stop_token >= 1 )
        {
            // Record error for this stream
            if( stream_id >= 0 && stream_id < MAX_STREAMS )
            {
                atomicExch( &g_stream_error_flags[stream_id], 1 );
            }
            __threadfence_system();

            // Trigger a stream error via nullptr dereference
            // This causes cudaStreamSynchronize to return cudaErrorIllegalAddress
            volatile int* null_ptr = nullptr;
            volatile int dummy = *null_ptr;

            // Use the result to prevent optimization
            if( dummy > 0 )
            {
                atomicExch( &g_stop_token, 3 );
            }
        }
    }
}

// Host-side class for managing soft trap state
class SoftTrapManager
{
  private:
    int host_stop_token;
    int host_stream_errors[MAX_STREAMS];
    bool initialized;

  public:
    SoftTrapManager() : initialized( false )
    {
        initialize();
    }

    void initialize()
    {
        if( initialized )
            return;

        resetStopToken();
        initialized = true;
    }

    // Reset stop token and all error flags to 0
    void resetStopToken()
    {
        int zero = 0;
        cudaMemcpyToSymbol( g_stop_token, &zero, sizeof( int ) );

        int zeros[MAX_STREAMS] = { 0 };
        cudaMemcpyToSymbol( g_stream_error_flags, zeros, sizeof( zeros ) );

        host_stop_token = 0;
        for( int i = 0; i < MAX_STREAMS; i++ )
        {
            host_stream_errors[i] = 0;
        }
    }

    // Query if stop token has been set
    bool hasStopToken()
    {
        cudaMemcpyFromSymbol( &host_stop_token, g_stop_token, sizeof( int ) );
        return host_stop_token == 1;
    }

    // Launch check_ok kernel on a specific stream
    void launchCheckOk( cudaStream_t stream, int stream_id )
    {
        if( stream_id < 0 || stream_id >= MAX_STREAMS )
        {
            std::cerr << "Error: stream_id " << stream_id << " out of range [0, " << MAX_STREAMS << ")" << std::endl;
            return;
        }
        check_ok_kernel<<<1, 1, 0, stream>>>( stream_id );
    }

    // Check if a specific stream has detected an error
    bool getStreamErrorStatus( int stream_id )
    {
        if( stream_id < 0 || stream_id >= MAX_STREAMS )
        {
            return false;
        }

        cudaMemcpyFromSymbol(
            &host_stream_errors[stream_id], g_stream_error_flags, sizeof( int ), stream_id * sizeof( int ) );
        return host_stream_errors[stream_id] == 1;
    }

    // Get all stream error statuses at once
    void getAllStreamErrorStatuses( bool * statuses, int num_streams )
    {
        if( num_streams > MAX_STREAMS )
            num_streams = MAX_STREAMS;

        cudaMemcpyFromSymbol( host_stream_errors, g_stream_error_flags, num_streams * sizeof( int ) );

        for( int i = 0; i < num_streams; i++ )
        {
            statuses[i] = ( host_stream_errors[i] == 1 );
        }
    }

    bool isInitialized() const
    {
        return initialized;
    }
};

// Global helper functions for convenience
namespace SoftTrap
{
static SoftTrapManager g_manager;

inline void reset()
{
    g_manager.resetStopToken();
}

inline bool hasError()
{
    return g_manager.hasStopToken();
}

inline void launchCheckOk( cudaStream_t stream, int stream_id )
{
    g_manager.launchCheckOk( stream, stream_id );
}

inline bool streamHasError( int stream_id )
{
    return g_manager.getStreamErrorStatus( stream_id );
}

inline void getAllStreamErrors( bool * statuses, int num_streams )
{
    g_manager.getAllStreamErrorStatuses( statuses, num_streams );
}
}

#endif // CUDA_SOFT_TRAP_H
