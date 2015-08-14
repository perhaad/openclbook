/* System includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* OpenCL includes */
#include <CL/cl.h>

/* Utility functions */
#include "utils.h"
#include "bmp-utils.h"
#include "gold.h"

/* Filter for the convolution */
static float gaussianBlurFilter[25] = {
   1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
   4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
   7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
   4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
   1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f};
static const int filterWidth = 5;
static const int filterSize = 25*sizeof(float);

/* Number of histogram bins */
static const int HIST_BINS = 256; 

int main(int argc, char **argv) 
{
   /* Host data */
   float *hInputImage = NULL; 
   int *hOutputHistogram = NULL;

   /* Allocate space for the input image and read the
    * data from disk */
   int imageRows;
   int imageCols;
   hInputImage = readBmpFloat("../../Images/cat.bmp", &imageRows, &imageCols);
   const int imageElements = imageRows*imageCols;
   const size_t imageSize = imageElements*sizeof(float);

   /* Allocate space for the histogram on the host */
   const int histogramSize = HIST_BINS*sizeof(int);
   hOutputHistogram = (int*)malloc(histogramSize);
   if (!hOutputHistogram) { exit(-1); }

   /* Use this to check the output of each API call */
   cl_int status;

   /* Get the first platform */
   cl_platform_id platform;
   status = clGetPlatformIDs(1, &platform, NULL);
   check(status);

   /* Get the devices */
   cl_device_id devices[2];
   cl_device_id gpuDevice;
   cl_device_id cpuDevice;
   status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpuDevice, NULL);
   check(status);
   status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &cpuDevice, NULL);
   check(status);
   devices[0] = gpuDevice;
   devices[1] = cpuDevice;

   /* Create a context and associate it with the devices */
   cl_context context;
   context = clCreateContext(NULL, 2, devices, NULL, NULL, &status);
   check(status);

   /* Create the command queues */
   cl_command_queue gpuQueue;
   cl_command_queue cpuQueue;
   gpuQueue = clCreateCommandQueue(context, gpuDevice, 0, &status);
   check(status);
   cpuQueue = clCreateCommandQueue(context, cpuDevice, 0, &status);
   check(status);

   /* The image descriptor describes how the data will be stored 
    * in memory. This descriptor initializes a 2D image with no pitch */
   cl_image_desc desc;
   desc.image_type = CL_MEM_OBJECT_IMAGE2D;
   desc.image_width = imageCols;
   desc.image_height = imageRows;
   desc.image_depth = 0;
   desc.image_array_size = 0;
   desc.image_row_pitch = 0;
   desc.image_slice_pitch = 0;
   desc.num_mip_levels = 0;
   desc.num_samples = 0;
   desc.buffer = NULL;

   /* The image format describes the properties of each pixel */
   cl_image_format format;
   format.image_channel_order = CL_R; // single channel
   format.image_channel_data_type = CL_FLOAT;

   /* Create the input image and initialize it using a 
    * pointer to the image data on the host. */
   cl_mem inputImage;
   inputImage = clCreateImage(context, CL_MEM_READ_ONLY,
      &format, &desc, NULL, NULL);

   /* Create a buffer object for the output histogram */
   cl_mem outputHistogram;
   outputHistogram = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
      histogramSize, NULL, &status);
   check(status);

     /* Create a buffer for the filter */
   cl_mem filter;
   filter = clCreateBuffer(context, CL_MEM_READ_ONLY, filterSize, 
      NULL, &status);
   check(status);

   cl_mem pipe;
#ifdef OCL_PIPES
   assert(0);
#else
   pipe = clCreateBuffer(context, CL_MEM_READ_WRITE, imageSize, NULL, &status);
   check(status);
#endif

   /* Copy the host image data to the GPU */
   size_t origin[3] = {0, 0, 0}; // Offset within the image to copy from
   size_t region[3] = {imageCols, imageRows, 1}; // Elements to per dimension
   status = clEnqueueWriteImage(gpuQueue, inputImage, CL_TRUE, 
      origin, region, 0 /* row-pitch */, 0 /* slice-pitch */, 
      hInputImage, 0, NULL, NULL);
   check(status);

   /* Write the filter to the GPU */
   status = clEnqueueWriteBuffer(gpuQueue, filter, CL_TRUE, 0, 
      filterSize, gaussianBlurFilter, 0, NULL, NULL);
   check(status);

   /* Initialize the output histogram with zeros */
   int zero = 0;
   status = clEnqueueFillBuffer(cpuQueue, outputHistogram, &zero, 
      sizeof(int), 0, histogramSize, 0, NULL, NULL);
   check(status);

   /* Create a program with source code */
   char *programSource = readFile("producer-consumer.cl");
   size_t programSourceLen = strlen(programSource);
   cl_program program = clCreateProgramWithSource(context, 1,
      (const char**)&programSource, &programSourceLen, &status);
   check(status);

   /* Build (compile) the program for the devices */
   status = clBuildProgram(program, 2, devices, NULL, NULL, NULL);
   if (status != CL_SUCCESS) {
      printCompilerError(program, gpuDevice);
      exit(-1);
   }

   /* Create the kernels */
   cl_kernel producerKernel;
   cl_kernel consumerKernel;
   producerKernel = clCreateKernel(program, "producerKernel", &status);
   check(status);
   consumerKernel = clCreateKernel(program, "consumerKernel", &status);
   check(status);

   /* Set the kernel arguments */
#ifdef OCL_PIPES
   status  = clSetKernelArg(producerKernel, 0, sizeof(cl_mem), &inputImage);
   status |= clSetKernelArg(producerKernel, 1, sizeof(cl_mem), &pipe);
   status |= clSetKernelArg(producerKernel, 2, sizeof(cl_mem), &filter);
   status |= clSetKernelArg(producerKernel, 3, sizeof(int), &filterWidth);
   check(status);
#else
   status  = clSetKernelArg(producerKernel, 0, sizeof(cl_mem), &inputImage);
   status |= clSetKernelArg(producerKernel, 1, sizeof(int), &imageRows);
   status |= clSetKernelArg(producerKernel, 2, sizeof(int), &imageCols);
   status |= clSetKernelArg(producerKernel, 3, sizeof(cl_mem), &pipe);
   status |= clSetKernelArg(producerKernel, 4, sizeof(cl_mem), &filter);
   status |= clSetKernelArg(producerKernel, 5, sizeof(int), &filterWidth);
   check(status);
#endif

   status  = clSetKernelArg(consumerKernel, 0, sizeof(cl_mem), &pipe);
   status |= clSetKernelArg(consumerKernel, 1, sizeof(int), &imageElements);
   status |= clSetKernelArg(consumerKernel, 2, sizeof(cl_mem), &outputHistogram);
   check(status);

   /* Define the index space and work-group size */
   size_t producerGlobalSize[2];
   producerGlobalSize[0] = imageCols;
   producerGlobalSize[1] = imageRows;

   size_t producerLocalSize[2];
   producerLocalSize[0] = 8;
   producerLocalSize[1] = 8;

   size_t consumerGlobalSize[1];
   consumerGlobalSize[0] = 1;

   size_t consumerLocalSize[1];
   consumerLocalSize[0] = 1;

   /* Enqueue the kernels for execution */
   status = clEnqueueNDRangeKernel(gpuQueue, producerKernel, 2, NULL,
      producerGlobalSize, producerLocalSize, 0, NULL, NULL);
   check(status);

#ifndef OCL_PIPES
   /* If pipes aren't supported, we need to run sequentially */
   clFinish(gpuQueue);
#endif

   status = clEnqueueNDRangeKernel(cpuQueue, consumerKernel, 1, NULL,
      consumerGlobalSize, consumerLocalSize, 0, NULL, NULL);
   check(status);

   /* Read the output histogram buffer to the host */
   status = clEnqueueReadBuffer(cpuQueue, outputHistogram, CL_TRUE, 0,
         histogramSize, hOutputHistogram, 0, NULL, NULL);
   check(status);

   /* Verify the result */
   float *refConvolution = convolutionGoldFloat(hInputImage, 
      imageRows, imageCols, gaussianBlurFilter, filterWidth);
   int *refHistogram = histogramGoldFloat(refConvolution, imageRows*imageCols,
         HIST_BINS);
   int i;
   int passed = 1;
   for (i = 0; i < HIST_BINS; i++) {
      if (hOutputHistogram[i] != refHistogram[i]) {
         passed = 0;
      }
   }
   if (passed) {
      printf("Passed!\n");
   }
   else {
      printf("Failed.\n");
   }
   free(refConvolution);
   free(refHistogram);

   /* Free OpenCL resources */
   clReleaseKernel(producerKernel);
   clReleaseKernel(consumerKernel);
   clReleaseProgram(program);
   clReleaseCommandQueue(gpuQueue);
   clReleaseCommandQueue(cpuQueue);
   clReleaseMemObject(inputImage);
   clReleaseMemObject(outputHistogram);
   clReleaseMemObject(filter);
   clReleaseMemObject(pipe);
   clReleaseContext(context);

   /* Free host resources */
   free(hInputImage);
   free(hOutputHistogram);
   free(programSource);

   return 0;
}
