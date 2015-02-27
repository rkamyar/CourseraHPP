#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define TILE_WIDTH 12
#define BLOCK_WIDTH (TILE_WIDTH + 4)
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE

__global__ void convolution_2D_kernel(float *P,float *I, int height, int width, int channels, const float * __restrict__ M) 
{

	__shared__ float Ns[w][w];
	
	int k;
   for (k = 0; k < channels; k++) 
   {
	
		// loading batch
      int dest = threadIdx.y * TILE_WIDTH + threadIdx.x, destY = dest / w, destX = dest % w,srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius, srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius,
         src = (srcY * width + srcX) * channels + k;
      if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
         Ns[destY][destX] = I[src];
      else
         Ns[destY][destX] = 0;
	   
	   //load another batch
	   
	    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
		  destY = dest / w, destX = dest % w;
		  srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
		  srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
		  src = (srcY * width + srcX) * channels + k;
		  if (destY < w) 
		  {
			 if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
				Ns[destY][destX] = I[src];
			 else
				Ns[destY][destX] = 0;
		  }
	   __syncthreads();
	   
	   float output=0.0;
	    int y, x;
      for (y = 0; y < Mask_width; y++)
         for (x = 0; x < Mask_width; x++)
            output += Ns[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];
     
		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
      x = blockIdx.x * TILE_WIDTH + threadIdx.x;
      if (y < height && x < width)
         P[(y * width + x) * channels + k] = clamp(output);
	   
      __syncthreads();
	   
		   
   }
	
		
}
	
	
int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	
	dim3 dimGrid(ceil((float)imageWidth/TILE_WIDTH), ceil((float)imageHeight/TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	
	   
    convolution_2D_kernel<<<dimGrid, dimBlock>>>(deviceOutputImageData, deviceInputImageData, imageHeight,
                                                 imageWidth, imageChannels,
                                                 deviceMaskData);
	wbCheck( cudaThreadSynchronize() );
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
