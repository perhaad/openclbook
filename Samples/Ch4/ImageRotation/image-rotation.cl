__constant sampler_t sampler =
   CLK_NORMALIZED_COORDS_FALSE | 
   CLK_FILTER_LINEAR           |
   CLK_ADDRESS_CLAMP;

__kernel 
void rotation(
    __read_only image2d_t inputImage, 
   __write_only image2d_t outputImage,
                      int imageWidth,
                      int imageHeight,
                    float theta)
{
   /* Get global ID for output coordinates */
   int x = get_global_id(0);
   int y = get_global_id(1);

   /* Compute image center */
   float x0 = imageWidth/2.0f;
   float y0 = imageHeight/2.0f;

   /* Compute the work-item's location relative
    * to the image center */
   int xprime = x-x0;
   int yprime = y-y0;

   /* Compute sine and cosine */
   float sinTheta = sin(theta);
   float cosTheta = cos(theta);

   /* Compute the input location */
   float2 readCoord;
   readCoord.x = xprime*cosTheta - yprime*sinTheta + x0;
   readCoord.y = xprime*sinTheta + yprime*cosTheta + y0;

   /* Read the input image */
   float value;   
   value = read_imagef(inputImage, sampler, readCoord).x;

   /* Write the output image */
   write_imagef(outputImage, (int2)(x, y), (float4)(value, 0.f, 0.f, 0.f));

}
