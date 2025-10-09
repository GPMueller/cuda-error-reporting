#include "../cuda_soft_trap.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

// Vector addition kernel that checks for errors (e.g., NaN values)
__global__ void vector_add_with_check( const float * a, const float * b, float * c, int n, bool introduce_error )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < n )
    {
        float val_a = a[idx];
        float val_b = b[idx];

        // Check for NaN or introduce error condition
        if( isnan( val_a ) || isnan( val_b ) || introduce_error )
        {
            // Soft trap! This will set the global stop token
            if( threadIdx.x == 0 && blockIdx.x == 0 )
            {
                printf( "GPU: Error detected in vector_add at index %d! Calling softTrap()...\n", idx );
            }
            softTrap();
            return;
        }

        c[idx] = val_a + val_b;
    }
}

// Regular vector addition kernel (no error checking)
__global__ void vector_add_normal( const float * a, const float * b, float * c, int n )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < n )
    {
        c[idx] = a[idx] + b[idx];
    }
}

void printVector( const char * name, const float * vec, int n, int max_print = 10 )
{
    std::cout << name << ": [";
    for( int i = 0; i < std::min( n, max_print ); i++ )
    {
        std::cout << vec[i];
        if( i < std::min( n, max_print ) - 1 )
            std::cout << ", ";
    }
    if( n > max_print )
        std::cout << ", ...";
    std::cout << "]" << std::endl;
}

int main()
{
    std::cout << "=== CUDA Soft Trap Multi-Stream Example ===" << std::endl;
    std::cout << std::endl;

    const int N    = 1024;
    const int size = N * sizeof( float );

    // Allocate host memory
    float * h_a1 = new float[N];
    float * h_b1 = new float[N];
    float * h_c1 = new float[N];

    float * h_a2 = new float[N];
    float * h_b2 = new float[N];
    float * h_c2 = new float[N];

    // Initialize input vectors
    for( int i = 0; i < N; i++ )
    {
        h_a1[i] = 1.0f;
        h_b1[i] = 2.0f;
        h_c1[i] = 0.0f;

        h_a2[i] = 3.0f;
        h_b2[i] = 4.0f;
        h_c2[i] = 0.0f;
    }

    // Allocate device memory for stream 1
    float *d_a1, *d_b1, *d_c1;
    cudaMalloc( &d_a1, size );
    cudaMalloc( &d_b1, size );
    cudaMalloc( &d_c1, size );

    // Allocate device memory for stream 2
    float *d_a2, *d_b2, *d_c2;
    cudaMalloc( &d_a2, size );
    cudaMalloc( &d_b2, size );
    cudaMalloc( &d_c2, size );

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate( &stream1 );
    cudaStreamCreate( &stream2 );

    std::cout << "Created 2 CUDA streams" << std::endl;
    std::cout << "Stream 1 will process a vector addition that encounters an error" << std::endl;
    std::cout << "Stream 2 will process a normal vector addition concurrently" << std::endl;
    std::cout << std::endl;

    // Reset soft trap state
    SoftTrap::reset();

    // Copy data to device for both streams
    cudaMemcpyAsync( d_a1, h_a1, size, cudaMemcpyHostToDevice, stream1 );
    cudaMemcpyAsync( d_b1, h_b1, size, cudaMemcpyHostToDevice, stream1 );

    cudaMemcpyAsync( d_a2, h_a2, size, cudaMemcpyHostToDevice, stream2 );
    cudaMemcpyAsync( d_b2, h_b2, size, cudaMemcpyHostToDevice, stream2 );

    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid   = ( N + threadsPerBlock - 1 ) / threadsPerBlock;

    std::cout << "Launching kernels..." << std::endl;
    std::cout << "  Stream 1: vector_add_with_check (will error)" << std::endl;
    std::cout << "  Stream 2: vector_add_with_check (normal operation)" << std::endl;
    std::cout << std::endl;

    // Stream 1: Launch kernel that WILL encounter an error and call softTrap()
    vector_add_with_check<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>( d_a1, d_b1, d_c1, N, true );
    SoftTrap::launchCheckOk( stream1, 0 ); // Check after stream 1 kernel

    // Stream 2: Launch kernel that would normally succeed, but will be stopped by check_ok
    vector_add_with_check<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>( d_a2, d_b2, d_c2, N, false );
    SoftTrap::launchCheckOk( stream2, 1 ); // Check after stream 2 kernel

    // Synchronize streams
    cudaError_t err1 = cudaStreamSynchronize( stream1 );
    cudaError_t err2 = cudaStreamSynchronize( stream2 );

    std::cout << "Streams synchronized" << std::endl;
    std::cout << "  Stream 1 status: " << cudaGetErrorString( err1 ) << std::endl;
    std::cout << "  Stream 2 status: " << cudaGetErrorString( err2 ) << std::endl;
    std::cout << std::endl;

    // Check soft trap status
    bool stop_token_set = SoftTrap::hasError();
    std::cout << "Stop token set: " << ( stop_token_set ? "YES" : "NO" ) << std::endl;

    // Check individual stream errors
    bool stream1_error = SoftTrap::streamHasError( 0 );
    bool stream2_error = SoftTrap::streamHasError( 1 );

    std::cout << "Stream 1 error flag: " << ( stream1_error ? "YES" : "NO" ) << std::endl;
    std::cout << "Stream 2 error flag: " << ( stream2_error ? "YES" : "NO" ) << std::endl;
    std::cout << std::endl;

    if( stream1_error )
    {
        std::cout << "✓ Stream 1 detected error as expected" << std::endl;
    }
    if( stream2_error )
    {
        std::cout << "✓ Stream 2 also detected the stop token" << std::endl;
    }
    std::cout << std::endl;

    // Copy results back (may be incomplete due to error)
    cudaMemcpyAsync( h_c1, d_c1, size, cudaMemcpyDeviceToHost, stream1 );
    cudaMemcpyAsync( h_c2, d_c2, size, cudaMemcpyDeviceToHost, stream2 );
    cudaDeviceSynchronize();

    std::cout << "Results (may be incomplete due to soft trap):" << std::endl;
    printVector( "Stream 1 output", h_c1, N );
    printVector( "Stream 2 output", h_c2, N );
    std::cout << std::endl;

    // === Demonstrate context recovery ===
    std::cout << "=== Demonstrating Context Recovery ===" << std::endl;
    std::cout << "Resetting soft trap and launching successful kernel..." << std::endl;
    std::cout << std::endl;

    // Reset the soft trap state
    SoftTrap::reset();

    // Launch a successful kernel to prove context is still valid
    vector_add_normal<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>( d_a1, d_b1, d_c1, N );
    cudaError_t recovery_err = cudaStreamSynchronize( stream1 );

    std::cout << "Recovery kernel status: " << cudaGetErrorString( recovery_err ) << std::endl;

    if( recovery_err == cudaSuccess )
    {
        cudaMemcpy( h_c1, d_c1, size, cudaMemcpyDeviceToHost );
        std::cout << "✓ Context is still valid! Successfully computed vector addition." << std::endl;
        printVector( "Recovered output", h_c1, N );

        // Verify result
        bool correct = true;
        for( int i = 0; i < N; i++ )
        {
            if( fabs( h_c1[i] - 3.0f ) > 1e-5 )
            {
                correct = false;
                break;
            }
        }
        if( correct )
        {
            std::cout << "✓ Result is correct (1.0 + 2.0 = 3.0)" << std::endl;
        }
    }
    else
    {
        std::cout << "✗ Context appears to be corrupted" << std::endl;
    }

    // Cleanup
    cudaFree( d_a1 );
    cudaFree( d_b1 );
    cudaFree( d_c1 );
    cudaFree( d_a2 );
    cudaFree( d_b2 );
    cudaFree( d_c2 );

    cudaStreamDestroy( stream1 );
    cudaStreamDestroy( stream2 );

    delete[] h_a1;
    delete[] h_b1;
    delete[] h_c1;
    delete[] h_a2;
    delete[] h_b2;
    delete[] h_c2;

    std::cout << std::endl;
    std::cout << "=== Example Complete ===" << std::endl;

    return 0;
}
