#include "../cuda_global_error_handler.h"
#include <cuda_runtime.h>
#include <iostream>

// Kernel with no error reporter parameter
__global__ void simple_kernel( int * data, int size )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < size && data[idx] < 0 )
    {
        REPORT_ERROR( COMPUTATION_ERROR, "Found negative value" );
    }
}

int main()
{
    const int size = 100;

    std::cout << "=== Minimal Global Error Reporting Example ===" << std::endl;

    // Allocate device memory
    int * d_data;
    cudaMalloc( &d_data, size * sizeof( int ) );

    // Initialize test data with an error condition
    int * h_data = new int[size];
    for( int i = 0; i < size; i++ )
    {
        h_data[i] = i;
    }
    h_data[50] = -10; // Introduce an error

    cudaMemcpy( d_data, h_data, size * sizeof( int ), cudaMemcpyHostToDevice );

    // Launch kernel
    dim3 block( 256 );
    dim3 grid( ( size + block.x - 1 ) / block.x );

    simple_kernel<<<grid, block>>>( d_data, size );

    // Simple error checking - just like cudaGetLastError!
    CHECK_KERNEL_NAMED( "simple_kernel" );

    std::cout << "\nTesting without error..." << std::endl;

    // Reset data to good values
    for( int i = 0; i < size; i++ )
    {
        h_data[i] = i + 1;
    }
    cudaMemcpy( d_data, h_data, size * sizeof( int ), cudaMemcpyHostToDevice );

    simple_kernel<<<grid, block>>>( d_data, size );
    CHECK_KERNEL_NAMED( "simple_kernel (no error)" );

    std::cout << "No errors detected - success!" << std::endl;

    // Cleanup
    delete[] h_data;
    cudaFree( d_data );

    return 0;
}
