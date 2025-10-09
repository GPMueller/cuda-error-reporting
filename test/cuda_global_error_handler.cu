#include "../cuda_global_error_handler.h"
#include <cuda_runtime.h>
#include <iostream>

// Example kernel that might encounter errors - NO REPORTER PARAMETER NEEDED!
__global__ void example_kernel( float * data, int size )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < size )
    {
        float value = data[idx];

        // Check for potential error conditions
        if( value < 0.0f )
        {
            REPORT_ERROR( COMPUTATION_ERROR, "Negative value detected" );
            return;
        }

        if( value > 1000.0f )
        {
            REPORT_ERROR( COMPUTATION_ERROR, "Value too large" );
            return;
        }

        // Simulate a memory access error
        if( idx >= size - 10 && value > 900.0f )
        {
            REPORT_ERROR( MEMORY_ERROR, "Near boundary with large value" );
            return;
        }

        // Normal computation
        data[idx] = sqrtf( value * value + 1.0f );
    }
}

// Matrix kernel - also no reporter parameter needed
__global__ void matrix_kernel( float * matrix, int rows, int cols )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if( row < rows && col < cols )
    {
        int idx = row * cols + col;

        // Check for division by zero
        if( matrix[idx] == 0.0f && row == col )
        {
            REPORT_ERROR( COMPUTATION_ERROR, "Zero on diagonal - singular matrix" );
            return;
        }

        // Some matrix operation
        if( row == col )
        {
            matrix[idx] = 1.0f / matrix[idx];
        }
        else
        {
            matrix[idx] *= 2.0f;
        }
    }
}

// Simple validation kernel
__global__ void validation_kernel( int * data, int size )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < size )
    {
        if( data[idx] < 0 )
        {
            REPORT_ERROR( COMPUTATION_ERROR, "Invalid negative data" );
            return;
        }

        // Some processing
        data[idx] = data[idx] * 2 + 1;
    }
}

int main()
{
    const int size        = 1000;
    const int matrix_size = 10;

    // Initialize CUDA
    cudaSetDevice( 0 );

    // No need to create error reporters - the global system handles it!

    std::cout << "=== Global Error Reporting System Demo ===" << std::endl;
    std::cout << "No need to pass error reporters to kernels!" << std::endl << std::endl;

    // Allocate device memory
    float * d_vector;
    float * d_matrix;
    int * d_int_data;

    cudaMalloc( &d_vector, size * sizeof( float ) );
    cudaMalloc( &d_matrix, matrix_size * matrix_size * sizeof( float ) );
    cudaMalloc( &d_int_data, size * sizeof( int ) );

    // Initialize test data
    float * h_vector = new float[size];
    float * h_matrix = new float[matrix_size * matrix_size];
    int * h_int_data = new int[size];

    // Set up data with intentional errors
    for( int i = 0; i < size; i++ )
    {
        h_vector[i]   = static_cast<float>( i % 100 );
        h_int_data[i] = i;
        if( i == 500 )
            h_vector[i] = -5.0f; // Negative value
        if( i == 750 )
            h_vector[i] = 1500.0f; // Large value
        if( i == 995 )
            h_vector[i] = 950.0f; // Near boundary + large
        if( i == 300 )
            h_int_data[i] = -10; // Negative int
    }

    for( int i = 0; i < matrix_size * matrix_size; i++ )
    {
        h_matrix[i] = static_cast<float>( i + 1 );
        if( i == 5 * matrix_size + 5 )
            h_matrix[i] = 0.0f; // Zero on diagonal
    }

    // Copy data to device
    cudaMemcpy( d_vector, h_vector, size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_matrix, h_matrix, matrix_size * matrix_size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_int_data, h_int_data, size * sizeof( int ), cudaMemcpyHostToDevice );

    dim3 block_size( 256 );
    dim3 grid_size( ( size + block_size.x - 1 ) / block_size.x );

    std::cout << "=== Testing Vector Kernel ===" << std::endl;
    try
    {
        // Launch vector kernel - NO REPORTER PARAMETER!
        example_kernel<<<grid_size, block_size>>>( d_vector, size );
        CHECK_KERNEL();
        std::cout << "=== TEST PASSED ===" << std::endl;
    }
    catch( const CudaException & e )
    {
        std::cout << "TEST FAILED, caught an unexpected exception: " << e.what() << std::endl;
        std::terminate();
    }

    std::cout << "\n=== Testing Matrix Kernel ===" << std::endl;
    try
    {
        // Launch matrix kernel
        dim3 matrix_block( 16, 16 );
        dim3 matrix_grid(
            ( matrix_size + matrix_block.x - 1 ) / matrix_block.x,
            ( matrix_size + matrix_block.y - 1 ) / matrix_block.y );
        matrix_kernel<<<matrix_grid, matrix_block>>>( d_matrix, matrix_size, matrix_size );
        CHECK_KERNEL_NAMED( "matrix_kernel" );
        std::cout << "TEST FAILED, expected exception to be thrown" << std::endl;
        std::terminate();
    }
    catch( const CudaException & e )
    {
        std::cout << "CudaException: " << e.what() << std::endl;
        std::cout << "=== TEST PASSED ===" << std::endl;
    }

    std::cout << "\n=== Testing Validation Kernel ===" << std::endl;
    try
    {
        validation_kernel<<<grid_size, block_size>>>( d_int_data, size );
        CHECK_KERNEL_NAMED( "validation_kernel" );
        std::cout << "TEST FAILED, expected exception to be thrown" << std::endl;
        std::terminate();
    }
    catch( const CudaException & e )
    {
        std::cout << "CudaException: " << e.what() << std::endl;
        std::cout << "=== TEST PASSED ===" << std::endl;
    }

    std::cout << "\n=== Testing Clean Run (No Errors) ===" << std::endl;
    // Reset data to good values
    for( int i = 0; i < size; i++ )
    {
        h_vector[i]   = static_cast<float>( i % 50 + 1 ); // All positive, reasonable
        h_int_data[i] = i + 1;                            // All positive
    }
    cudaMemcpy( d_vector, h_vector, size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_int_data, h_int_data, size * sizeof( int ), cudaMemcpyHostToDevice );

    try
    {
        example_kernel<<<grid_size, block_size>>>( d_vector, size );
        CHECK_KERNEL_NAMED( "example_kernel (clean)" );
        std::cout << "=== TEST PASSED ===" << std::endl;
    }
    catch( const CudaException & e )
    {
        std::cout << "TEST FAILED, caught an unexpected exception: " << e.what() << std::endl;
        std::terminate();
    }

    try
    {
        validation_kernel<<<grid_size, block_size>>>( d_int_data, size );
        CHECK_KERNEL_NAMED( "validation_kernel (clean)" );
        std::cout << "=== TEST PASSED ===" << std::endl;
    }
    catch( const CudaException & e )
    {
        std::cout << "TEST FAILED, caught an unexpected exception: " << e.what() << std::endl;
        std::terminate();
    }

    std::cout << "\n=== Testing Asynchronous Error Detection ===" << std::endl;
    try
    {
        // Launch kernel without waiting
        h_int_data[50] = -25;
        cudaMemcpy( d_int_data, h_int_data, size * sizeof( int ), cudaMemcpyHostToDevice );

        validation_kernel<<<grid_size, block_size>>>( d_int_data, size );

        std::cout << "Kernel launched, doing other work..." << std::endl;

        // Simulate other work
        for( volatile int i = 0; i < 1000000; i++ )
        {
            // Busy wait
        }

        std::cout << "Checking for errors..." << std::endl;
        CHECK_KERNEL_NAMED( "validation_kernel (async)" );
        std::cout << "TEST FAILED, expected exception to be thrown" << std::endl;
        std::terminate();
    }
    catch( const CudaException & e )
    {
        std::cout << "CudaException: " << e.what() << std::endl;
        std::cout << "=== TEST PASSED ===" << std::endl;
    }

    std::cout << "\n=== Testing Multiple Sequential Kernels ===" << std::endl;
    try
    {
        // Launch multiple kernels and check each one
        for( int i = 1; i <= 3; i++ )
        {
            // Create different error conditions
            for( int j = 0; j < size; j++ )
            {
                h_int_data[j] = j;
            }
            if( i == 2 )
                h_int_data[i * 100] = -i; // Only second kernel will have error

            cudaMemcpy( d_int_data, h_int_data, size * sizeof( int ), cudaMemcpyHostToDevice );

            std::cout << "Launching kernel " << i << "..." << std::endl;
            validation_kernel<<<grid_size, block_size>>>( d_int_data, size );

            std::string kernel_name = "validation_kernel_" + std::to_string( i );
            CHECK_KERNEL_NAMED( kernel_name.c_str() );
        }
        std::cout << "TEST FAILED, expected exception to be thrown" << std::endl;
        std::terminate();
    }
    catch( const CudaException & e )
    {
        std::cout << "CudaException: " << e.what() << std::endl;
        std::cout << "=== TEST PASSED ===" << std::endl;
    }

    // Cleanup
    delete[] h_vector;
    delete[] h_matrix;
    delete[] h_int_data;
    cudaFree( d_vector );
    cudaFree( d_matrix );
    cudaFree( d_int_data );

    std::cout << "\n=== Demo completed successfully! ===" << std::endl;

    return 0;
}
