#define __CL_ENABLE_EXCEPTIONS

#include <fstream>
#include <iostream>
#include <vector>

#include <CL/cl.hpp>

#include "utils.h"
#include "bmp-utils.h"
#include "gold.h"

static const char* inputImagePath = "../../Images/cat.bmp";

static float gaussianBlurFilterFactor = 273.0f;
static float gaussianBlurFilter[25] = {
   1.0f,  4.0f,  7.0f,  4.0f, 1.0f,
   4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
   7.0f, 26.0f, 41.0f, 26.0f, 7.0f,
   4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
   1.0f,  4.0f,  7.0f,  4.0f, 1.0f};
static const int gaussianBlurFilterWidth = 5;

static float sharpenFilterFactor = 8.0f;
static float sharpenFilter[25] = {
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f,  2.0f,  2.0f,  2.0f, -1.0f,
    -1.0f,  2.0f,  8.0f,  2.0f, -1.0f,
    -1.0f,  2.0f,  2.0f,  2.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
static const int sharpenFilterWidth = 5;

static float edgeSharpenFilterFactor = 1.0f;
static float edgeSharpenFilter[9] = {
    1.0f,  1.0f, 1.0f,
    1.0f, -7.0f, 1.0f,
    1.0f,  1.0f, 1.0f};
static const int edgeSharpenFilterWidth = 3;

static float vertEdgeDetectFilterFactor = 1.0f;
static float vertEdgeDetectFilter[25] = {
     0,  0, -1.0f,  0,  0,
     0,  0, -1.0f,  0,  0,
     0,  0,  4.0f,  0,  0,
     0,  0, -1.0f,  0,  0,
     0,  0, -1.0f,  0,  0};
static const int vertEdgeDetectFilterWidth = 3;

static float embossFilterFactor = 1.0f;
static float embossFilter[9] = {
    2.0f,  0.0f,  0.0f,
    0.0f, -1.0f,  0.0f,
    0.0f,  0.0f, -1.0f};
static const int embossFilterWidth = 3;

enum filterList 
{
   GAUSSIAN_BLUR,
   SHARPEN, 
   EDGE_SHARPEN, 
   VERT_EDGE_DETECT, 
   EMBOSS, 
   FILTER_LIST_SIZE
};
static const int filterSelection = VERT_EDGE_DETECT;

int main() 
{
   float *hInputImage;
   float *hOutputImage;

   int imageRows;
   int imageCols;

   /* Set the filter here */
   int filterWidth;
   float filterFactor;
   float *filter;

   switch (filterSelection) 
   {
    case GAUSSIAN_BLUR:
      filterWidth = gaussianBlurFilterWidth;
      filterFactor = gaussianBlurFilterFactor;
      filter = gaussianBlurFilter;
      break;
    case SHARPEN:
      filterWidth = sharpenFilterWidth;
      filterFactor = sharpenFilterFactor;
      filter = sharpenFilter;
      break;
    case EDGE_SHARPEN:
      filterWidth = edgeSharpenFilterWidth;
      filterFactor = edgeSharpenFilterFactor;
      filter = edgeSharpenFilter;
      break;
    case VERT_EDGE_DETECT:
      filterWidth = vertEdgeDetectFilterWidth;
      filterFactor = vertEdgeDetectFilterFactor;
      filter = vertEdgeDetectFilter;
      break;
    case EMBOSS:
      filterWidth = embossFilterWidth;
      filterFactor = embossFilterFactor;
      filter = embossFilter;
      break;
    default:  
      std::cout << "Invalid filter selection." << std::endl;
      return 1;
   }

   for (int i = 0; i < filterWidth*filterWidth; i++) 
   {
      filter[i] = filter[i]/filterFactor;
   }


   /* Read in the BMP image */
   hInputImage = readBmpFloat(inputImagePath, &imageRows, &imageCols);

   /* Allocate space for the output image */
   hOutputImage = new float [imageRows*imageCols];

   try 
   {
      /* Query for platforms */
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);

      /* Get a list of devices on this platform */
      std::vector<cl::Device> devices;
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      /* Create a context for the devices */
      cl::Context context(devices);
      
      /* Create a command queue for the first device */
      cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

      /* Create the images */
      cl::ImageFormat imageFormat = cl::ImageFormat(CL_R, CL_FLOAT);
      cl::Image2D inputImage = cl::Image2D(context, CL_MEM_READ_ONLY,
           imageFormat, imageCols, imageRows);
      cl::Image2D outputImage = cl::Image2D(context, CL_MEM_WRITE_ONLY,
           imageFormat, imageCols, imageRows);
      
      /* Create a buffer for the filter */
      cl::Buffer filterBuffer = cl::Buffer(context, CL_MEM_READ_ONLY,
           filterWidth*filterWidth*sizeof(float));
      
      /* Copy the input data to the input image */
      cl::size_t<3> origin;
      origin[0] = 0;
      origin[1] = 0;
      origin[2] = 0;
      cl::size_t<3> region;
      region[0] = imageCols;
      region[1] = imageRows;
      region[2] = 1;
      queue.enqueueWriteImage(inputImage, CL_TRUE, origin, region, 0, 0,
           hInputImage);

      /* Copy the filter to the buffer */
      queue.enqueueWriteBuffer(filterBuffer, CL_TRUE, 0,
           filterWidth*filterWidth*sizeof(float), filter);

      /* Create the sampler */
      cl::Sampler sampler = cl::Sampler(context, CL_FALSE, 
         CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);
      
      /* Read the program source */
      std::ifstream sourceFile("image-convolution.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));
      
      /* Make program from the source code */
      cl::Program program = cl::Program(context, source);
      
      /* Build the program for the devices */
      program.build(devices);
      
      /* Create the kernel */
      cl::Kernel kernel(program, "convolution");
      
      /* Set the kernel arguments */
      kernel.setArg(0, inputImage);
      kernel.setArg(1, outputImage);
      kernel.setArg(2, filterBuffer);
      kernel.setArg(3, filterWidth);
      kernel.setArg(4, sampler);
      
      /* Execute the kernel */
      cl::NDRange global(imageCols, imageRows);
      cl::NDRange local(8, 8);
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
      
      /* Copy the output data back to the host */
      queue.enqueueReadImage(outputImage, CL_TRUE, origin, region, 0, 0,
           hOutputImage);

      /* Save the output bmp */
      writeBmpFloat(hOutputImage, "cat-filtered.bmp", imageRows, imageCols,
           inputImagePath);
   }
   catch(cl::Error error)
   {
      std::cout << error.what() << "(" << error.err() << ")" << std::endl;
   }

   /* Verify result */
   float *refOutput = convolutionGoldFloat(hInputImage, imageRows, imageCols,
      filter, filterWidth);
   int i;
   bool passed = true;
   for (i = 0; i < imageRows*imageCols; i++) {
      if (abs(refOutput[i]-hOutputImage[i]) > 0.001f) {
         passed = false;
      }
   }
   if (passed) {
      std::cout << "Passed!" << std::endl;
   }
   else {
      std::cout << "Failed." << std::endl;
   }
   free(refOutput);

   free(hInputImage);
   delete hOutputImage;
   return 0;
}
