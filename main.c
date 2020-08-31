/**
 * LIBRARIES
 * */
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <tiffio.h>
#include <math.h>
#include <time.h>
#include <zconf.h>
#pragma OPENCL EXTENSION cl_intel_printf : enable
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

/**
 * CONSTANTS
 * */
#define MAX_SOURCE_SIZE (0x100000)
#define PI 3.14159265358979323846
#define znew  ((z = 36969 * (z & 65535) + (z >> 16)) << 16)
#define wnew  ((w = 18000 * (w & 65535) + (w >> 16)) & 65535)
#define MWC   (znew + wnew)
#define SHR3  (jsr = (jsr = (jsr = jsr ^ (jsr << 17)) ^ (jsr >> 13)) ^ (jsr << 5))
#define CONG  (jcong = 69069 * jcong + 1234567)
#define KISS  ((MWC^CONG)+SHR3)

struct coordinate {
    double x;
    double y;
};

typedef struct coordinate Coordinate;

double serpentineR, lemniscateR, permutationR, permutationX0, xsinR;
Coordinate lemniscateC0, lemniscateC1, xsinC0;
uint32 permutationM, lemniscateM, xsinM;
uint32 signature;
unsigned int z, w, jsr, jcong;

double xcrt[5] , ycrt[5] , pxcrt[5] , pycrt[5];
unsigned int sump[4] , iv , cp = 0;

/**
 * LOGGING
 * */
const char *getErrorString(cl_int error) {
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

void bin(unsigned int n) {
    unsigned i;
    for (i = 1 << 31; i > 0; i = i / 2)
        (n & i)? printf("1"): printf("0");
    printf("\n");
}

void logParameters() {
    printf("Parameters used: \n\n");

    printf("Function used for permutation [ARCTAN(CTG)]:\n");
    printf("\tpermutationR: %0.15lf\n", permutationR);
    printf("\tpermutationM: %u\n", permutationM);
    printf("\tpermutationX0: %0.15lf\n\n", permutationX0);

    printf("Function used for nonce 1[LEMNISCATE]:\n");
    printf("\tlemniscateR: %0.15lf\n", lemniscateR);
    printf("\tlemniscateM: %u\n", lemniscateM);
    printf("\tlemniscateC0: ( %0.15lf, %0.15lf)\n\n", lemniscateC0.x, lemniscateC0.y);

    printf("Function used for XOR operation [SERPENTINE]: \n");
    printf("\tserpentineR: %0.15lf \n", serpentineR);

    printf("Function used for pixels' interdependence [XSIN]:\n");
    printf("\txsinR: %0.15lf\n", xsinR);
    printf("\txsinM: %u\n", xsinM);
    printf("\txsinC0: ( %0.15lf, %0.15lf)\n\n", xsinC0.x, xsinC0.y);

}

/**
 * RANDOM GENERATORS
 * */
uint32 pseudoIRandom(uint32 start, uint32 end) {
    double r = (double) rand() / RAND_MAX;
    return (uint32)(start + r * (end - start));
}

double pseudoDRandom(double start, double end) {
    double r = (double) rand() / RAND_MAX;
    return start + r * (end - start);
}

/**
 * AUXILIARY FUNCTIONS
 * */
unsigned int greatestPowerOf2Divisor(unsigned int number) {
    unsigned int pwr = 0;

    while(CL_TRUE) {
        ++pwr;
        unsigned int aux = pow(2, pwr);

        if(number % aux || aux > number)
            return --pwr;

        if ((uint32)pow(2,pwr) == 64) {
            return pwr;
        }
    };

    return pwr;
}

void incrementLemniscateMap() {
    uint32 i;
    double lemniscate2Pwr = pow(2, lemniscateR);

    for (i = 0; i < lemniscateM; ++i) { // increment lemniscate map
        lemniscateC0.x = cos(lemniscateR) / ( 1 + pow(sin(lemniscate2Pwr * lemniscateC0.y), 2.0));
        lemniscateC0.y = (2 * sqrt(2) * sin(lemniscate2Pwr * lemniscateC0.x) * cos(lemniscate2Pwr * lemniscateC0.x)) *
                         ( 1 + pow(sin(lemniscate2Pwr * lemniscateC0.x), 2.0));
    }

    lemniscateC1.x = lemniscateC0.x;
    lemniscateC1.y = lemniscateC0.y;
}

void incrementXsinxMap() {
    uint32 i;
    double xsinwPwr = pow(3, xsinR);
    double aux;

    for (i = 0; i < xsinM; ++i) {
        aux = xsinC0.x;
        xsinC0.x = atan2(1.0, tan(xsinC0.x + sin(xsinwPwr * xsinC0.y)));
        xsinC0.y = aux;
    }
}

/**
 * ENCRYPTION/DECRYPTION
 * */
void performXOR(uint32* inputImage, uint32 imageSize) {
    uint32 i;
    double lemniscate2Pwr = pow(2, lemniscateR);

    double decimalsPwr = pow(10, 15), serpentine2Pwr = pow(2, serpentineR);
    unsigned int aux;
    unsigned long long XORseq;
    unsigned int decimals =  floor(decimalsPwr * fabs(lemniscateC1.y));
    XORseq = decimals ^ (unsigned long long) floor(( 1.0 / ( imageSize)) * decimalsPwr);
    aux = floor(decimalsPwr * fabs( atan( 1.0 / tan( decimalsPwr * (double) XORseq))));
    aux = aux << 8u;
    aux = aux >> 8u;
    signature = (unsigned int)aux ^ imageSize;

    // Firstly we open the kernel file and we store its contents into a string
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("image_XOR.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);


    cl_platform_id platforms[32];
    cl_uint platformsNum;
    cl_context_properties contextProperties[3];
    cl_platform_id fastestPlatform = 0;
    cl_device_id fastestDevice = 0;
    cl_uint flopsMax = 0;
    int j;

    cl_int ret = clGetPlatformIDs((cl_uint)32, platforms, &platformsNum);

    for(i = 0; i < platformsNum; ++i) {
        cl_device_id *devices;
        cl_uint devicesCount, maxClockFreq, computeUnitsNum;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &devicesCount);
        devices = (cl_device_id*)malloc(sizeof(cl_device_id) * devicesCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, devicesCount, devices, NULL);

        for (j = 0; j < devicesCount; ++j) {
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            sizeof(cl_uint), &maxClockFreq, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(cl_uint), &computeUnitsNum, NULL);
            if(flopsMax < (maxClockFreq * computeUnitsNum)) {
                fastestPlatform = platforms[i];
                fastestDevice = devices[j];
                flopsMax = maxClockFreq * computeUnitsNum;
            }
        }
        free(devices); devices = NULL;
    }
    contextProperties[0] = (cl_context_properties)CL_CONTEXT_PLATFORM;
    contextProperties[1] = (cl_context_properties)fastestPlatform;
    contextProperties[2] =(cl_context_properties)0;

    size_t localItemSize = (uint32)pow(2, greatestPowerOf2Divisor(imageSize)); // Divide work items into groups of 64

    // Create an OpenCL context
    cl_context context = clCreateContext( contextProperties, 1, &fastestDevice, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, fastestDevice, 0, &ret);

    // Create memory buffers on the device for each vector
    cl_mem inputImgBuff = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      imageSize * sizeof(uint32), NULL, &ret);
    cl_mem outputImgBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      imageSize * sizeof(uint32), NULL, &ret);

    cl_mem rtBuff = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                          sizeof(double), NULL, &ret);
    cl_mem nonceBuff = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                         sizeof(double), NULL, &ret);

    cl_mem signatureBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         sizeof(uint32), NULL, &ret);
    cl_mem tempBuff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     sizeof(uint32) * localItemSize, NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(commandQueue, inputImgBuff, CL_TRUE, 0,
                               imageSize * sizeof(uint32), inputImage, 0, NULL, NULL);

    ret = clEnqueueWriteBuffer(commandQueue, rtBuff, CL_TRUE, 0,
                               sizeof(double), &serpentineR, 0, NULL, NULL);

    ret = clEnqueueWriteBuffer(commandQueue, nonceBuff, CL_TRUE, 0,
                               sizeof(double), &(lemniscateC1.y), 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &fastestDevice, NULL, NULL, NULL);

    if (ret != CL_SUCCESS) {
        size_t len;
        char buffer[204800];
        cl_build_status bldstatus;
        printf("\nError %d: Failed to build program executable [ %s ]\n",ret,getErrorString(ret));
        ret = clGetProgramBuildInfo(program, fastestDevice, CL_PROGRAM_BUILD_STATUS, sizeof(bldstatus), (void *)&bldstatus, &len);
        if (ret != CL_SUCCESS)
        {
            printf("Build Status error %d: %s\n",ret,getErrorString(ret));
            exit(1);
        }
        if (bldstatus == CL_BUILD_SUCCESS) printf("Build Status: CL_BUILD_SUCCESS\n");
        if (bldstatus == CL_BUILD_NONE) printf("Build Status: CL_BUILD_NONE\n");
        if (bldstatus == CL_BUILD_ERROR) printf("Build Status: CL_BUILD_ERROR\n");
        if (bldstatus == CL_BUILD_IN_PROGRESS) printf("Build Status: CL_BUILD_IN_PROGRESS\n");
        ret = clGetProgramBuildInfo(program, fastestDevice, CL_PROGRAM_BUILD_OPTIONS, sizeof(buffer), buffer, &len);
        if (ret != CL_SUCCESS)
        {
            printf("Build Options error %d: %s\n",ret,getErrorString(ret));
            exit(1);
        }
        printf("Build Options: %s\n", buffer);
        ret = clGetProgramBuildInfo(program, fastestDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        if (ret != CL_SUCCESS)
        {
            printf("Build Log error %d: %s\n",ret,getErrorString(ret));
            exit(1);
        }
        printf("Build Log:\n%s\n", buffer);
        exit(1);
    }


    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "image_XOR", &ret);

    cl_double spr2Pwr = serpentine2Pwr;

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputImgBuff);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputImgBuff);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_double), (void *)&decimalsPwr);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_double), (void *)&spr2Pwr);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_uint), (void *)&decimals);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&signatureBuff);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = imageSize; // Process the entire lists

    ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                 &global_item_size, &localItemSize, 0, NULL, NULL);

    // Read the memory buffer C on the device to the local variable C

    ret = clEnqueueReadBuffer(commandQueue, outputImgBuff, CL_TRUE, 0,
                              imageSize * sizeof(uint32), inputImage, 0, NULL, NULL);
    uint32 textSignature;
    ret = clEnqueueReadBuffer(commandQueue, signatureBuff, CL_TRUE, 0,
                              sizeof(uint32), &textSignature, 0, NULL, NULL);

    signature = signature ^ imageSize ^ textSignature;

    // Clean up
    ret = clFlush(commandQueue);
    ret = clFinish(commandQueue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(inputImgBuff);
    ret = clReleaseMemObject(outputImgBuff);
    ret = clReleaseMemObject(rtBuff);
    ret = clReleaseMemObject(nonceBuff);
    ret = clReleaseMemObject(signatureBuff);
    ret = clReleaseCommandQueue(commandQueue);
    ret = clReleaseContext(context);

}

void encrypt(uint32* inputImage, uint32 width, uint32 height) {
    performXOR(inputImage, width * height);

    uint32 IV1 = floor(pow(10, 15) * xsinC0.x), IV2 = floor(pow(10, 15) * xsinC0.y);
    uint32 i;

    inputImage[0] = inputImage[0] ^ (IV1 << (IV2 % 11));

    for (i = 1; i < width * height; ++i) {
        inputImage[i] ^= (inputImage[i - 1] << 7u);
    }
}

void decrypt(uint32* inputImage, uint32 width, uint32 height) {

    uint32 IV1 = floor(pow(10, 15) * (xsinC0.x)), IV2 = floor(pow(10, 15) * xsinC0.y);
    uint32 i;

    for (i = width * height - 1; i >= 1; --i) {
        inputImage[i] ^= (inputImage[i - 1] << 7u);
    }

    inputImage[0] = inputImage[0] ^ (IV1 << (IV2 % 11));

    performXOR(inputImage, width * height);
}

/**
 * INIT DATA
 * */

void initializeKey(const char* MODE) {
    int keyDescriptor;

    if (!strcmp(MODE, "ENCRYPT")) {
        srand( time ( NULL) );

        serpentineR = pseudoDRandom(10, 110);
        lemniscateR = pseudoDRandom(250, 350);
        permutationR = pseudoDRandom(PI/2, 10);
        xsinR = pseudoDRandom(10, 110);
        permutationX0 = pseudoDRandom(-1, 1);
        lemniscateC0.x = pseudoDRandom(-1, 1);
        lemniscateC0.y = pseudoDRandom(-1, 1);
        xsinC0.x = pseudoDRandom(-PI/2, PI/2);
        xsinC0.y = pseudoDRandom(-PI/2, PI/2);
        permutationM = pseudoIRandom(1000, 1100);
        lemniscateM = pseudoIRandom(1000, 1100);
        xsinM = pseudoIRandom(1000, 1100);

        //write key
        if ((keyDescriptor = open("/home/razvan.paraschiv/Desktop/OpenCL_test/key.txt", O_CREAT | O_WRONLY | O_TRUNC, S_IRWXU)) == -1) {
            perror("Error creating key file:");
        }

        write(keyDescriptor, &serpentineR, sizeof(serpentineR));
        write(keyDescriptor, &lemniscateR, sizeof(lemniscateR));
        write(keyDescriptor, &permutationR, sizeof(permutationR));
        write(keyDescriptor, &permutationX0, sizeof(permutationX0));
        write(keyDescriptor, &lemniscateC0, sizeof(lemniscateC0));
        write(keyDescriptor, &permutationM, sizeof(permutationM));
        write(keyDescriptor, &lemniscateM, sizeof(lemniscateM));
        write(keyDescriptor, &xsinC0, sizeof(xsinC0));
        write(keyDescriptor, &xsinM, sizeof(xsinM));
        write(keyDescriptor, &xsinR, sizeof(xsinR));

        return;
    }

    if ((keyDescriptor = open("/home/razvan.paraschiv/Desktop/OpenCL_test/key.txt", O_RDONLY)) == -1) {
        perror("Error reading key file:");
    }

    /**
     * In order to test key sensitivity
     * manually add changes below
     * */
    read(keyDescriptor, &serpentineR, sizeof(serpentineR));
    read(keyDescriptor, &lemniscateR, sizeof(lemniscateR));
    read(keyDescriptor, &permutationR, sizeof(permutationR));
    read(keyDescriptor, &permutationX0, sizeof(permutationX0));
    read(keyDescriptor, &lemniscateC0, sizeof(lemniscateC0));
    read(keyDescriptor, &permutationM, sizeof(permutationM));
    read(keyDescriptor, &lemniscateM, sizeof(lemniscateM));
    read(keyDescriptor, &xsinC0, sizeof(xsinC0));
    read(keyDescriptor, &xsinM, sizeof(xsinM));
    read(keyDescriptor, &xsinR, sizeof(xsinR));


    close(keyDescriptor);
}

/**
 * PERMUTATION
 * */
uint32 *getPermutation(uint32 imageSize, const char* mode) {

    uint32 i;
    uint32 max = imageSize;

    uint32 * permutation = (uint32*) malloc(sizeof(uint32*) * imageSize);
    cl_bool *labelArray = (cl_bool *) calloc(imageSize, sizeof(cl_bool));

    for(i = 0; i < imageSize; ++i) {
        labelArray[i] = CL_FALSE;
    }

    for(i = 0; i < permutationM; ++i) {
        permutationX0 = ( 2 * atan2( 1.0 , tan(permutationR * permutationX0)))/ PI;
    }

    for (i = 0; i < imageSize; ++i) {

        permutation[i] = (unsigned long long)floor(pow(10, 15) * fabs(permutationX0)) % imageSize;

        if (labelArray[permutation[i]]) {
            uint32 j = max - 1;
            while (j >= 0 && labelArray[j]) {
                --j;
            }
            max = j;
            permutation[i] = max;
        }
        labelArray[permutation[i]] = CL_TRUE;

        permutationX0 = ( 2 * atan( 1.0 / tan(permutationR * permutationX0)))/ PI;
    }

    free(labelArray);

    if (!strcmp(mode, "SHUFFLE"))
        return permutation;

    uint32 * inversePermutation = (uint32*)malloc(sizeof(uint32*) * imageSize);

    for (i = 0; i < imageSize; ++i) {
        inversePermutation[permutation[i]] = i;
    }

    free(permutation);

    return inversePermutation;
}

void permute(uint32** inputImage, uint32 imageSize, const char* mode) {
    uint32* permutation = getPermutation(imageSize, mode);
    uint32 *permutedImage = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));
    uint32 i;

    for (i = 0; i < imageSize; ++i) {
        permutedImage[i] = (*inputImage)[permutation[i]];
    }
    _TIFFfree(*inputImage);
    *inputImage = permutedImage;

    free(permutation);
}

void shuffle(uint32** inputImage, uint32 imageSize) {
    permute(inputImage, imageSize,"SHUFFLE");
}

void unshuffle(uint32** inputImage, uint32 imageSize) {
    permute(inputImage, imageSize, "UNSHUFFLE");
}

/**
 * PROCESS IMAGE
 * */
void convertImage(TIFF* inputTiff, TIFF* outputTiff, const char * mode) {

    initializeKey(mode);

    logParameters();

    uint32 width, height;
    short bitsPerSample, samplePixel, orientation, photometric, planarConfig;
    uint16 extrasamples;

    TIFFGetField(inputTiff, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(inputTiff, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(inputTiff, TIFFTAG_SAMPLESPERPIXEL, &samplePixel);
    TIFFGetField(inputTiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFSetField(outputTiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFGetField(inputTiff, TIFFTAG_PLANARCONFIG, &planarConfig);
    TIFFGetField(inputTiff, TIFFTAG_PHOTOMETRIC, &photometric);

    TIFFSetField(outputTiff, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(outputTiff, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(outputTiff, TIFFTAG_SAMPLESPERPIXEL, samplePixel);
    TIFFSetField(outputTiff, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
    TIFFSetField(outputTiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(outputTiff, TIFFTAG_PLANARCONFIG, planarConfig);
    TIFFSetField(outputTiff, TIFFTAG_PHOTOMETRIC, photometric);

    uint32 imageSize = width * height;
    uint32 *inputImage = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));

    TIFFReadRGBAImage(inputTiff, width, height, inputImage, 0);

    tsize_t lineBytes = 4 * width;
    uint32 * lineBuffer = NULL;

    if (TIFFScanlineSize(outputTiff) == lineBytes) {
        lineBuffer = (uint32 *)_TIFFmalloc(lineBytes);
    }
    else {
        lineBuffer = (uint32 *)_TIFFmalloc(TIFFScanlineSize(outputTiff));
    }

    TIFFSetField(outputTiff, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(outputTiff, width * samplePixel));

    incrementLemniscateMap();
    incrementXsinxMap();

    if (!strcmp(mode, "ENCRYPT")) {
        // start clock permutation
        clock_t permutationClockBegin = clock();

        shuffle(&inputImage, imageSize);

        clock_t permutationClockEnd = clock();
        double shuffleTimeSpent = (double)(permutationClockEnd - permutationClockBegin) / CLOCKS_PER_SEC;

        printf("Time spent for shuffling: %lf\n", shuffleTimeSpent);

        // start clock encryption
        clock_t encryptionClockBegin = clock();

        encrypt(inputImage, width, height);

        clock_t encryptionClockEnd = clock();
        double cryptTimeSpent = (double)(encryptionClockEnd - encryptionClockBegin) / CLOCKS_PER_SEC;

        printf("Time spent for encryption: %lf\n", cryptTimeSpent);

        printf("Total time spent: %lf\n", shuffleTimeSpent + cryptTimeSpent);
    } else {
        // start clock encryption
        clock_t encryptionClockBegin = clock();

        decrypt(inputImage, width, height);

        clock_t encryptionClockEnd = clock();
        double cryptTimeSpent = (double)(encryptionClockEnd - encryptionClockBegin) / CLOCKS_PER_SEC;

        printf("Time spent for decryption: %lf\n", cryptTimeSpent);

        // start clock permutation
        clock_t permutationClockBegin = clock();

        unshuffle(&inputImage, imageSize);

        clock_t permutationClockEnd = clock();
        double shuffleTimeSpent = (double)(permutationClockEnd - permutationClockBegin) / CLOCKS_PER_SEC;

        printf("Time spent for unshuffling: %lf\n", shuffleTimeSpent);

        printf("Total time spent: %lf\n", shuffleTimeSpent + cryptTimeSpent);
    }

    // write image to new TIFF
    uint32 row;
    for (row = 0; row < height; ++row) {
        memcpy(lineBuffer, inputImage + height * (width - row - 1), lineBytes);    // check the index here, and figure out why not using h*linebytes
        if (TIFFWriteScanline(outputTiff, lineBuffer, row, 0) < 0)
            break;
    }

    _TIFFfree(lineBuffer);
    _TIFFfree(inputImage);
}

int main(int argc, char** argv) {

    if (argc != 4) {
        fprintf(stderr, "Usage format: %s <MODE> <INPUT> <OUTPUT>\n", argv[0]);
        return 1;
    }

    if(strcmp(argv[1], "ENCRYPT") != 0 && strcmp(argv[1], "DECRYPT") != 0) {
        errno = EINVAL;
        perror("Invalid MODE parameter:");
    }
    char *inp = (char*)malloc(sizeof(char) * PATH_MAX);
    char *out = (char*)malloc(sizeof(char) * PATH_MAX);
    const char* dir =  "demo/";
    strcat(inp, dir);
    strcat(inp, argv[2]);
    strcat(out, dir);
    strcat(out, argv[3]);

    // Open Tiff files
    TIFF *inputTiff = TIFFOpen(inp, "r"),
        *outputTiff = TIFFOpen(out, "w");

    if (!inputTiff || !outputTiff) {
        perror("Problem while reading/writing tiff:");
    }

    convertImage(inputTiff, outputTiff, argv[1]);

    free(inp);
    free(out);
    TIFFClose(inputTiff);
    TIFFClose(outputTiff);

    return EXIT_SUCCESS;
}


// Functions use for cryptanalysis
double imageEntropy(TIFF* img) {
    unsigned int **frequency = (unsigned int**)calloc(256,sizeof(unsigned int*));
    for(unsigned int i=0;i<256;i++)
        frequency[i] = (unsigned int*)calloc(3,sizeof(unsigned int));

    unsigned int *total = (unsigned int*)calloc(3,sizeof(unsigned int));
    uint32 width, height;
    TIFFGetField(img, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(img, TIFFTAG_IMAGELENGTH, &height);

    uint32 imageSize = width * height;
    uint32 *inputImage = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));

    TIFFReadRGBAImage(img, width, height, inputImage, 0);

    for( int j=0 ; j < width ; j++)
        for( int i=0 ; i < height ; i++)
        {
            unsigned char aux_r = (inputImage[i * width + j] & 0x000000ff);
            frequency[aux_r][0]++;
            total[0]++;

            unsigned char aux_g = ((inputImage[i * width + j] >> 8u)  & 0x000000ff);
            frequency[aux_g][1]++;
            total[1]++;

            unsigned char aux_b = ((inputImage[i * width + j] >> 16u) & 0x000000ff);
            frequency[aux_b][2]++;
            total[2]++;
        }

    double *entropy = (double*)calloc(3,sizeof(double));

    for(unsigned int j=0;j<3;j++)
    {
        for(unsigned int i=0;i<256;i++)
            if(frequency[i][j]!=0)
            {
                double p = (double)frequency[i][j]/total[j];
                entropy[j] = entropy[j] + p*(log(p)/log(2));
            }
        entropy[j] = -entropy[j];
    }
    _TIFFfree(inputImage);
    return (entropy[0] + entropy[1] + entropy[2])/3;
}

double imageHistogramUniformity(TIFF * img) {
    unsigned int **frequency = (unsigned int**)calloc(256,sizeof(unsigned int*));
    for(unsigned int i=0;i<256;i++)
        frequency[i] = (unsigned int*)calloc(3,sizeof(unsigned int));

    uint32 width, height;
    TIFFGetField(img, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(img, TIFFTAG_IMAGELENGTH, &height);

    uint32 imageSize = width * height;
    uint32 *inputImage = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));

    TIFFReadRGBAImage(img, width, height, inputImage, 0);

    for( int j=0 ; j < height ; j++)
        for( int i=0 ; i < width ; i++)
        {
            frequency[inputImage[i * width + j] & 0x000000ff][0]++;

            frequency[ (inputImage[i * width + j] >> 8u)  & 0x000000ff][1]++;

            frequency[(inputImage[i * width + j] >> 16u) & 0x000000ff][2]++;
        }

    double *chi_square = (double*)calloc(3,sizeof(double));

    double estimated = height * width/256.0;

    for(unsigned int j=0;j<3;j++)
        for( int i=0 ; i < 256 ; i++)
            chi_square[j] = chi_square[j] + pow(frequency[i][j] - estimated,2)/estimated;

    _TIFFfree(inputImage);

    return (chi_square[0] + chi_square[1] + chi_square[2])/3;
}

double imageUACI(TIFF* img1, TIFF* img2) {
    double *UACI = (double *)calloc(3,sizeof(double));

    uint32 width, height;
    TIFFGetField(img1, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(img1, TIFFTAG_IMAGELENGTH, &height);

    uint32 imageSize = width * height;
    uint32 *inputImage1 = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));
    uint32 *inputImage2 = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));

    TIFFReadRGBAImage(img1, width, height, inputImage1, 0);
    TIFFReadRGBAImage(img2, width, height, inputImage2, 0);

    for(unsigned int j = 0; j < height; j++)
        for( unsigned int i = 0; i < width; i++)
        {
            UACI[0] = UACI[0] + abs((inputImage1[i * width + j] & 0x000000ff) - (inputImage2[i * width + j] & 0x000000ff));
            UACI[1] = UACI[1] + abs(((inputImage1[i * width + j] >> 8u)  & 0x000000ff) - ((inputImage2[i * width + j] >> 8u)  & 0x000000ff));
            UACI[2] = UACI[2] + abs(((inputImage1[i * width + j] >> 16u)  & 0x000000ff) - ((inputImage2[i * width + j] >> 16u)  & 0x000000ff));
        }

    for(unsigned int i=0;i<3;i++)
        UACI[i] = (UACI[i]/(width*height*255))*100;

    _TIFFfree(inputImage1);
    _TIFFfree(inputImage2);
    return (UACI[0] + UACI[1] + UACI[2])/3;
}

double imageNPCR(TIFF* img1, TIFF* img2) {
    double *NPCR = (double *)calloc(3,sizeof(double));

    uint32 width, height;
    TIFFGetField(img1, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(img1, TIFFTAG_IMAGELENGTH, &height);

    uint32 imageSize = width * height;
    uint32 *inputImage1 = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));
    uint32 *inputImage2 = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));

    TIFFReadRGBAImage(img1, width, height, inputImage1, 0);
    TIFFReadRGBAImage(img2, width, height, inputImage2, 0);

    for(unsigned int j = 0; j < height; j++)
        for( unsigned int i = 0; i < width; i++)
        {
            NPCR[0] = NPCR[0] + (((inputImage1[i * width + j] & 0x000000ff) == (inputImage2[i * width + j] & 0x000000ff)) ? 0 : 1);
            NPCR[1] = NPCR[1] + ((((inputImage1[i * width + j] >> 8u)  & 0x000000ff) == ((inputImage2[i * width + j] >> 8u)  & 0x000000ff)) ? 0 : 1);
            NPCR[2] = NPCR[2] + ((((inputImage1[i * width + j] >> 16u) & 0x000000ff) == ((inputImage2[i * width + j] >> 16u) & 0x000000ff)) ? 0 : 1);
        }

    for(unsigned int i=0;i<3;i++)
        NPCR[i] = (NPCR[i]/(width*height))*100;

    _TIFFfree(inputImage1);
    _TIFFfree(inputImage2);

    return (NPCR[0] + NPCR[1] + NPCR[2])/3;;
}

double imageMSE(TIFF* inputImg, TIFF* outputImg) {
    double *MSE = (double *)calloc(3,sizeof(double));

    uint32 width, height;
    TIFFGetField(inputImg, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(inputImg, TIFFTAG_IMAGELENGTH, &height);

    uint32 imageSize = width * height;
    uint32 *inputImage = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));
    uint32 *outputImage = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));

    TIFFReadRGBAImage(inputImg, width, height, inputImage, 0);
    TIFFReadRGBAImage(outputImg, width, height, outputImage, 0);

    for(unsigned int j = 0; j < height; j++)
        for( unsigned int i = 0; i < width; i++)
        {
            MSE[0] = MSE[0] + pow((inputImage[i * width + j] & 0x000000ff)-(outputImage[i * width + j] & 0x000000ff),2);
            MSE[1] = MSE[1] + pow(((inputImage[i * width + j] >> 8u)  & 0x000000ff)-((outputImage[i * width + j] >> 8u)  & 0x000000ff), 2);
            MSE[2] = MSE[2] + pow(((inputImage[i * width + j] >> 16u)  & 0x000000ff)-((outputImage[i * width + j] >> 16u)  & 0x000000ff), 2);
        }

    for(unsigned int i=0;i<3;i++)
        MSE[i] = MSE[i]/(width * height);

    return (MSE[0] + MSE[1] + MSE[2])/3;
}

double imageCorrelationCoefficient(TIFF* inputImg, TIFF* outputImg) {
    double *e_in = (double *)calloc(3,sizeof(double));
    double *d_in = (double *)calloc(3,sizeof(double));
    double *e_out = (double *)calloc(3,sizeof(double));
    double *d_out = (double *)calloc(3,sizeof(double));
    double *cov = (double *)calloc(3,sizeof(double));

    uint32 width, height;
    TIFFGetField(inputImg, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(inputImg, TIFFTAG_IMAGELENGTH, &height);

    uint32 imageSize = width * height;
    uint32 *inputImage = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));
    uint32 *outputImage = (uint32*)_TIFFmalloc(imageSize * sizeof(uint32));

    TIFFReadRGBAImage(inputImg, width, height, inputImage, 0);
    TIFFReadRGBAImage(outputImg, width, height, outputImage, 0);

    for(unsigned int j = 0; j < height; j++)
        for( unsigned int i = 0; i < width; i++)
        {
            e_in[0] = e_in[0] + (inputImage[i * width + j]  & 0x000000ff);
            e_out[0] = e_out[0] + (outputImage[i * width + j]  & 0x000000ff);

            e_in[1] = e_in[1] + ((inputImage[i * width + j] >> 8u)  & 0x000000ff);
            e_out[1] = e_out[1] + ((outputImage[i * width + j] >> 8u)  & 0x000000ff);

            e_in[2] = e_in[2] + ((inputImage[i * width + j] >> 16u)  & 0x000000ff);
            e_out[2] = e_out[2] + ((outputImage[i * width + j] >> 16u)  & 0x000000ff);
        }

    for(unsigned int i=0;i<3;i++)
    {
        e_in[i] = e_in[i]/(width*height);
        e_out[i] = e_out[i]/(width*height);
    }

    for(unsigned int j = 0; j < height; j++)
        for( unsigned int i = 0; i < width; i++)
        {
            d_in[0] = d_in[0] + pow((inputImage[i * width + j]  & 0x000000ff)-e_in[0],2);
            d_out[0] = d_out[0] + pow((inputImage[i * width + j]  & 0x000000ff)-e_out[0],2);
            cov[0] = cov[0] + ((inputImage[i * width + j]  & 0x000000ff) - e_in[0])*((outputImage[i * width + j]  & 0x000000ff) - e_out[0]);

            d_in[1] = d_in[1] + pow(((inputImage[i * width + j] >> 8u)  & 0x000000ff)-e_in[1],2);
            d_out[1] = d_out[1] + pow(((outputImage[i * width + j] >> 8u)  & 0x000000ff)-e_out[1],2);
            cov[1] = cov[1] + (((inputImage[i * width + j] >> 8u)  & 0x000000ff) - e_in[1])*(((outputImage[i * width + j] >> 8u)  & 0x000000ff) - e_out[1]);

            d_in[2] = d_in[2] + pow(((inputImage[i * width + j] >> 16u)  & 0x000000ff)-e_in[2],2);
            d_out[2] = d_out[2] + pow(((outputImage[i * width + j] >> 16u)  & 0x000000ff)-e_out[2],2);
            cov[2] = cov[2] + (((inputImage[i * width + j] >> 16u)  & 0x000000ff) - e_in[2])*(((outputImage[i * width + j] >> 16u)  & 0x000000ff) - e_out[2]);
        }

    for(unsigned int i=0;i<3;i++)
    {
        d_in[i] = d_in[i]/(height*width);
        d_out[i] = d_out[i]/(height*width);
        cov[i] = cov[i]/(height*width);
    }

    double *cc = (double *)calloc(3,sizeof(double));
    for(unsigned int i=0;i<3;i++)
        cc[i] = cov[i]/(sqrt(d_in[i]*d_out[i]));

    return (cc[0] + cc[1] + cc[2])/3;
}

unsigned int RGB_Gray(unsigned char r , unsigned char g , unsigned char b)
{
    return (unsigned int)(0.2125*r + 0.7154*g + 0.0721*b);
    //return (unsigned int)((r + g + b) / 3.0);
}

double *imageCorrelationCoefficientAdjacentPixels(TIFF* inputImg, char *nume_csv) {
    srand(time(NULL));

    z=rand();
    w=rand();
    jsr=rand();
    jcong=rand();

    double *cc = (double *)calloc(3,sizeof(double));

    unsigned int imgHeight = 100;
    unsigned int  imgWidth = 100;

    short bitPerSample;

    TIFFGetField(inputImg, TIFFTAG_BITSPERSAMPLE, &bitPerSample);

    TIFF *img1, *img2, *img3, *img4;

    TIFFSetField(img1, TIFFTAG_IMAGEWIDTH, imgWidth);
    TIFFSetField(img1, TIFFTAG_IMAGELENGTH, imgHeight);
    TIFFSetField(img1, TIFFTAG_BITSPERSAMPLE, bitPerSample);

    TIFFSetField(img2, TIFFTAG_IMAGEWIDTH, imgWidth);
    TIFFSetField(img2, TIFFTAG_IMAGELENGTH, imgHeight);
    TIFFSetField(img1, TIFFTAG_BITSPERSAMPLE, bitPerSample);

    TIFFSetField(img3, TIFFTAG_IMAGEWIDTH, imgWidth);
    TIFFSetField(img3, TIFFTAG_IMAGELENGTH, imgHeight);
    TIFFSetField(img1, TIFFTAG_BITSPERSAMPLE, bitPerSample);

    TIFFSetField(img4, TIFFTAG_IMAGEWIDTH, imgWidth);
    TIFFSetField(img4, TIFFTAG_IMAGELENGTH, imgHeight);
    TIFFSetField(img1, TIFFTAG_BITSPERSAMPLE, bitPerSample);

    uint32 *inputImage = (uint32*)_TIFFmalloc(10000 * sizeof(uint32));
    uint32 *img1Raster = (uint32*)_TIFFmalloc(10000 * sizeof(uint32));
    uint32 *img2Raster = (uint32*)_TIFFmalloc(10000 * sizeof(uint32));
    uint32 *img3Raster = (uint32*)_TIFFmalloc(10000 * sizeof(uint32));
    uint32 *img4Raster = (uint32*)_TIFFmalloc(10000 * sizeof(uint32));

    TIFFReadRGBAImage(inputImg, 100, 100, inputImage, 0);

    for(unsigned int j = 0; j < imgHeight; j++)
        for( unsigned int i = 0; i < imgWidth; i++)
        {
            unsigned int rcrt = 1 + KISS % (imgHeight - 1);
            unsigned int ccrt = 1 + KISS % (imgWidth - 1);
            img1Raster[j * imgWidth + i] = inputImage[ccrt * imgWidth + rcrt];
            img2Raster[j * imgWidth + i] = inputImage[ccrt * imgWidth + rcrt + 1];
            img3Raster[j * imgWidth + i] = inputImage[(ccrt + 1) * imgWidth + rcrt];
            img4Raster[j * imgWidth + i] = inputImage[(ccrt + 1) * imgWidth + rcrt + 1];
        }

    cc[0] = imageCorrelationCoefficient(img1,img2);
    cc[1] = imageCorrelationCoefficient(img1,img3);
    cc[2] = imageCorrelationCoefficient(img1,img4);

    FILE *fout = fopen(nume_csv,"w");
    fprintf(fout,"P(x)(y),P(x+1)(y),P(x)(y+1)P(x+1)(y+1)\n");

    for(unsigned int j = 0; j < imgHeight; j++)
        for( unsigned int i = 0; i < imgWidth; i++)
        {
            fprintf(fout,"%u,",RGB_Gray(
                    (img1Raster[i * imgHeight + j]  & 0x000000ff),
                    ((img1Raster[i * imgHeight + j] >> 8u)  & 0x000000ff),
                    ((img1Raster[i * imgHeight + j] >> 16u)  & 0x000000ff)
                    ));
            fprintf(fout,"%u,",RGB_Gray(
                    (img2Raster[i * imgHeight + j]  & 0x000000ff),
                    ((img2Raster[i * imgHeight + j] >> 8u)  & 0x000000ff),
                    ((img2Raster[i * imgHeight + j] >> 16u)  & 0x000000ff)
                    ));
            fprintf(fout,"%u,",RGB_Gray(
                    (img3Raster[i * imgHeight + j]  & 0x000000ff),
                    ((img3Raster[i * imgHeight + j] >> 8u)  & 0x000000ff),
                    ((img3Raster[i * imgHeight + j] >> 16u)  & 0x000000ff)
                    ));
            fprintf(fout,"%u,",RGB_Gray(
                    (img4Raster[i * imgHeight + j]  & 0x000000ff),
                    ((img4Raster[i * imgHeight + j] >> 8u)  & 0x000000ff),
                    ((img4Raster[i * imgHeight + j] >> 16u)  & 0x000000ff)
                    ));
            fprintf(fout,"\n");
        }

    fclose(fout);
    _TIFFfree(inputImg);
    _TIFFfree(img1Raster);
    _TIFFfree(img2Raster);
    _TIFFfree(img3Raster);
    _TIFFfree(img4Raster);

    return cc;
}