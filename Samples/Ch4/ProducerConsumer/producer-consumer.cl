__constant sampler_t sampler =
   CLK_NORMALIZED_COORDS_FALSE | 
   CLK_FILTER_NEAREST          |
   CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel
void producerKernel(
   image2d_t __read_only inputImage,
#ifdef OCL_PIPES
   pipe __write_only float *outputPipe,
#else
   int rows,
   int cols,
   __global float* outputPipe,      
#endif
    __constant float* filter,
    int filterWidth)
{
   /* Store each work-item’s unique row and column */
   int column = get_global_id(0);
   int row = get_global_id(1);
   
   /* Half the width of the filter is needed for indexing
    * memory later */
   int halfWidth = (int)(filterWidth/2);
   
   /* Used to hold the value of the output pixel */
   float sum = 0.0f;
   
   /* Iterator for the filter */
   int filterIdx = 0;
   
   /* Each work-item iterates around its local area based on the
    * size of the filter */
   int2 coords; // Coordinates for accessing the image
   
   /* Iterate the filter rows */
   for(int i = -halfWidth; i <= halfWidth; i++) 
   {
      coords.y = row + i;
      /* Iterate over the filter columns */
      for(int j = -halfWidth; j <= halfWidth; j++) 
      {
         coords.x = column + j;
         
         /* Read a pixel from the image. A single channel image
          * stores the pixel in the ’x’ coordinate of the returned
          * vector. */
         float4 pixel;         
         pixel = read_imagef(inputImage, sampler, coords);
         sum += pixel.x * filter[filterIdx++];
      }
   }
   
   /* Write the output pixel to the pipe */
#ifdef OCL_PIPES
   write_pipe(outputPipe, &sum);
#else
   int gid = row*cols+column;
   outputPipe[gid] = sum;
#endif
}

__kernel
void consumerKernel(
#ifdef OCL_PIPES
   pipe __read_only float *inputPipe,
#else
   __global float *inputPipe,
#endif
   int totalPixels,
   __global int *histogram)
{
   int pixelCnt;
   float pixel;
   
   /* Loop to process all pixels from the producer kernel */
   for (pixelCnt = 0; pixelCnt < totalPixels; pixelCnt++)
   {
      
#ifdef OCL_PIPES
      /* Keep trying to read a pixel from the pipe 
       * until one becomes available */
      while(read_pipe(inputPipe, &pixel));
#else
      pixel = inputPipe[pixelCnt];
#endif
      
      /* Add the pixel value to the histogram */
      histogram[(int)pixel]++;
   }
}
