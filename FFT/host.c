#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"

const char* program_source = "\n"\
"#define mask_left fft_index                                                                          \n" \
"#define mask_right stage                                                                             \n" \
"#define shift_pos N2                                                                                 \n" \
"#define angle size                                                                                   \n" \
"#define start br.s0                                                                                  \n" \
"#define cosine x3.s0                                                                                 \n" \
"#define sine x3.s1                                                                                   \n" \
"#define wk x2                                                                                        \n" \
"                                                                                                     \n" \
"__kernel void fft_init(__global float2* g_data, __local float2* l_data,                              \n" \
"                       ulong points_per_group, ulong size, int dir) {                                \n" \
"                                                                                                     \n" \
"   ulong4 br, index;                                                                                 \n" \
"   ulong points_per_item, g_addr, l_addr, i, fft_index, stage, N2;                                   \n" \
"   float2 x1, x2, x3, x4, sum12, diff12, sum34, diff34;                                              \n" \
"                                                                                                     \n" \
"   points_per_item = points_per_group/get_local_size(0);                                             \n" \
"   l_addr = get_local_id(0) * points_per_item;                                                       \n" \
"   g_addr = get_group_id(0) * points_per_group + l_addr;                                             \n" \
"                                                                                                     \n" \
"   /* Load data from bit-reversed addresses and perform 4-point FFTs */                              \n" \
"   for(i=0; i<points_per_item; i+=4) {                                                               \n" \
"      index = (ulong4)(g_addr, g_addr+1, g_addr+2, g_addr+3);                                        \n" \
"      mask_left = size/2;                                                                            \n" \
"      mask_right = 1;                                                                                \n" \
"      shift_pos = (ulong)log2((float)size)-1;                                                        \n" \
"      br = (index << shift_pos) & mask_left;                                                         \n" \
"      br |= (index >> shift_pos) & mask_right;                                                       \n" \
"                                                                                                     \n" \
"      /* Bit-reverse addresses */                                                                    \n" \
"      while(shift_pos > 1) {                                                                         \n" \
"         shift_pos -= 2;                                                                             \n" \
"         mask_left >>= 1;                                                                            \n" \
"         mask_right <<= 1;                                                                           \n" \
"         br |= (index << shift_pos) & mask_left;                                                     \n" \
"         br |= (index >> shift_pos) & mask_right;                                                    \n" \
"      }                                                                                              \n" \
"                                                                                                     \n" \
"      /* Load global data */                                                                         \n" \
"      x1 = g_data[br.s0];                                                                            \n" \
"      x2 = g_data[br.s1];                                                                            \n" \
"      x3 = g_data[br.s2];                                                                            \n" \
"      x4 = g_data[br.s3];                                                                            \n" \
"                                                                                                     \n" \
"      sum12 = x1 + x2;                                                                               \n" \
"      diff12 = x1 - x2;                                                                              \n" \
"      sum34 = x3 + x4;                                                                               \n" \
"      diff34 = (float2)(x3.s1 - x4.s1, x4.s0 - x3.s0) * dir;                                         \n" \
"      l_data[l_addr] = sum12 + sum34;                                                                \n" \
"      l_data[l_addr+1] = diff12 + diff34;                                                            \n" \
"      l_data[l_addr+2] = sum12 - sum34;                                                              \n" \
"      l_data[l_addr+3] = diff12 - diff34;                                                            \n" \
"      l_addr += 4;                                                                                   \n" \
"      g_addr += 4;                                                                                   \n" \
"   }                                                                                                 \n" \
"                                                                                                     \n" \
"   /* Perform initial stages of the FFT - each of length N2*2 */                                     \n" \
"   for(N2 = 4; N2 < points_per_item; N2 <<= 1) {                                                     \n" \
"      l_addr = get_local_id(0) * points_per_item;                                                    \n" \
"      for(fft_index = 0; fft_index < points_per_item; fft_index += 2*N2) {                           \n" \
"         x1 = l_data[l_addr];                                                                        \n" \
"         l_data[l_addr] += l_data[l_addr + N2];                                                      \n" \
"         l_data[l_addr + N2] = x1 - l_data[l_addr + N2];                                             \n" \
"         for(i=1; i<N2; i++) {                                                                       \n" \
"            cosine = cos(M_PI_F*i/N2);                                                               \n" \
"            sine = dir * sin(M_PI_F*i/N2);                                                           \n" \
"            wk = (float2)(l_data[l_addr+N2+i].s0*cosine + l_data[l_addr+N2+i].s1*sine,               \n" \
"                          l_data[l_addr+N2+i].s1*cosine - l_data[l_addr+N2+i].s0*sine);              \n" \
"            l_data[l_addr+N2+i] = l_data[l_addr+i] - wk;                                             \n" \
"            l_data[l_addr+i] += wk;                                                                  \n" \
"         }                                                                                           \n" \
"         l_addr += 2*N2;                                                                             \n" \
"      }                                                                                              \n" \
"   }                                                                                                 \n" \
"   barrier(CLK_LOCAL_MEM_FENCE);                                                                     \n" \
"                                                                                                     \n" \
"   /* Perform FFT with other items in group - each of length N2*2 */                                 \n" \
"   stage = 2;                                                                                        \n" \
"   for(N2 = points_per_item; N2 < points_per_group; N2 <<= 1) {                                      \n" \
"      start = (get_local_id(0) + (get_local_id(0)/stage)*stage) * (points_per_item/2);               \n" \
"      angle = start % (N2*2);                                                                        \n" \
"      for(i=start; i<start + points_per_item/2; i++) {                                               \n" \
"         cosine = cos(M_PI_F*angle/N2);                                                              \n" \
"         sine = dir * sin(M_PI_F*angle/N2);                                                          \n" \
"         wk = (float2)(l_data[N2+i].s0*cosine + l_data[N2+i].s1*sine,                                \n" \
"                       l_data[N2+i].s1*cosine - l_data[N2+i].s0*sine);                               \n" \
"         l_data[N2+i] = l_data[i] - wk;                                                              \n" \
"         l_data[i] += wk;                                                                            \n" \
"         angle++;                                                                                    \n" \
"      }                                                                                              \n" \
"      stage <<= 1;                                                                                   \n" \
"      barrier(CLK_LOCAL_MEM_FENCE);                                                                  \n" \
"   }                                                                                                 \n" \
"                                                                                                     \n" \
"   /* Store results in global memory */                                                              \n" \
"   l_addr = get_local_id(0) * points_per_item;                                                       \n" \
"   g_addr = get_group_id(0) * points_per_group + l_addr;                                             \n" \
"   for(i=0; i<points_per_item; i+=4) {                                                               \n" \
"      g_data[g_addr] = l_data[l_addr];                                                               \n" \
"      g_data[g_addr+1] = l_data[l_addr+1];                                                           \n" \
"      g_data[g_addr+2] = l_data[l_addr+2];                                                           \n" \
"      g_data[g_addr+3] = l_data[l_addr+3];                                                           \n" \
"      g_addr += 4;                                                                                   \n" \
"      l_addr += 4;                                                                                   \n" \
"   }                                                                                                 \n" \
"}                                                                                                    \n" \
"                                                                                                     \n" \
"__kernel void fft_stage(__global float2* g_data, ulong stage, ulong points_per_group, int dir) {     \n" \
"                                                                                                     \n" \
"   ulong points_per_item, addr, N, ang, i;                                                           \n" \
"   float c, s;                                                                                       \n" \
"   float2 input1, input2, w;                                                                         \n" \
"                                                                                                     \n" \
"   points_per_item = points_per_group/get_local_size(0);                                             \n" \
"   addr = (get_group_id(0) + (get_group_id(0)/stage)*stage) * (points_per_group/2) +                 \n" \
"            get_local_id(0) * (points_per_item/2);                                                   \n" \
"   N = points_per_group*(stage/2);                                                                   \n" \
"   ang = addr % (N*2);                                                                               \n" \
"                                                                                                     \n" \
"   for(i=addr; i<addr + points_per_item/2; i++) {                                                    \n" \
"      c = cos(M_PI_F*ang/N);                                                                         \n" \
"      s = dir * sin(M_PI_F*ang/N);                                                                   \n" \
"      input1 = g_data[i];                                                                            \n" \
"      input2 = g_data[i+N];                                                                          \n" \
"      w = (float2)(input2.s0*c + input2.s1*s, input2.s1*c - input2.s0*s);                            \n" \
"      g_data[i] = input1 + w;                                                                        \n" \
"      g_data[i+N] = input1 - w;                                                                      \n" \
"      ang++;                                                                                         \n" \
"   }                                                                                                 \n" \
"}                                                                                                    \n" \
"                                                                                                     \n" \
"__kernel void fft_scale(__global float2* g_data, ulong points_per_group, ulong scale) {              \n" \
"                                                                                                     \n" \
"   ulong points_per_item, addr, i;                                                                   \n" \
"                                                                                                     \n" \
"   points_per_item = points_per_group/get_local_size(0);                                             \n" \
"   addr = get_group_id(0) * points_per_group + get_local_id(0) * points_per_item;                    \n" \
"                                                                                                     \n" \
"   for(i=addr; i<addr + points_per_item; i++) {                                                      \n" \
"      g_data[i] /= scale;                                                                            \n" \
"   }                                                                                                 \n" \
"}                                                                                                    \n";

const char* TranslateOpenCLError(cl_int errorCode)
{
    switch (errorCode)
    {
    case CL_SUCCESS:                            return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
    case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
    case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
    case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
    case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
    case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
    case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
    case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
    case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
    case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
    case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

    default:
        return "UNKNOWN ERROR CODE";
    }
}

// TODO print error to std ERROR

cl_uint getVersion(const char* versionName)
{
    if (strstr(versionName, "OpenCL 1.0") != NULL)
        return 10;
    if (strstr(versionName, "OpenCL 1.1") != NULL)
        return 11;
    if (strstr(versionName, "OpenCL 2.0") != NULL)
        return 20;
    if (strstr(versionName, "OpenCL 2.1") != NULL)
        return 21;
    if (strstr(versionName, "OpenCL 2.2") != NULL)
        return 22;
    return 0;
}

cl_int get_platforms(struct Platform* platforms, cl_uint num_platforms)
{
    if (num_platforms == 0)
        return 1;

    cl_platform_id* platforms_id;
    cl_device_id* devices_id;

    size_t stringLength;
    char* versionName;
    cl_int err;


    platforms_id = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms_id, NULL);
    if (CL_SUCCESS != err)
    {
        printf("Error: clGetplatform_ids() to get platforms returned %s.\n", TranslateOpenCLError(err));
        return 1;
    }

    for (cl_uint i = 0; i < num_platforms; i++)
    {
        // get platform name
        err = clGetPlatformInfo(platforms_id[i], CL_PLATFORM_NAME, 0, NULL, &stringLength);
        if (CL_SUCCESS != err)
        {
            printf("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", TranslateOpenCLError(err));
            return 1;
        }

        platforms[i].name = (cl_uchar*)malloc(stringLength * sizeof(cl_uchar));
        err = clGetPlatformInfo(platforms_id[i], CL_PLATFORM_NAME, stringLength, platforms[i].name, NULL);
        if (CL_SUCCESS != err)
        {
            printf("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME '%s'.\n", TranslateOpenCLError(err));
            return 1;
        }

        // get platform version
        err = clGetPlatformInfo(platforms_id[i], CL_PLATFORM_VERSION, 0, NULL, &stringLength);
        if (CL_SUCCESS != err)
        {
            printf("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
            return 1;
        }
        versionName = (char*)malloc(stringLength * sizeof(char));
        err = clGetPlatformInfo(platforms_id[i], CL_PLATFORM_VERSION, stringLength, versionName, NULL);
        if (CL_SUCCESS != err)
        {
            printf("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION '%s'.\n", TranslateOpenCLError(err));
            return 1;
        }
        platforms[i].version = getVersion(versionName);
        free(versionName);

        // get platform's devices
        err = clGetDeviceIDs(platforms_id[i], CL_DEVICE_TYPE_ALL, 0, NULL, &platforms[i].num_devices);
        if (CL_SUCCESS != err)
        {
            printf("clGetDeviceIDs() to get CL_DEVICE_TYPE_ALL length %s.\n", TranslateOpenCLError(err));
            return 1;
        }
        platforms[i].devices = (struct Device*)malloc(platforms[i].num_devices * sizeof(struct Device));

        devices_id = (cl_device_id*)malloc(platforms[i].num_devices * sizeof(cl_device_id));
        err = clGetDeviceIDs(platforms_id[i], CL_DEVICE_TYPE_ALL, platforms[i].num_devices, devices_id, NULL);
        if (CL_SUCCESS != err)
        {
            printf("clGetDeviceIDs() to get CL_DEVICE_TYPE_ALL %s.\n", TranslateOpenCLError(err));
            return 1;
        }

        for (cl_uint j = 0; j < platforms[i].num_devices; ++j)
        {
            platforms[i].devices[j].id = devices_id[j];

            // get device name
            err = clGetDeviceInfo(devices_id[j], CL_DEVICE_NAME, 0, NULL, &stringLength);
            if (CL_SUCCESS != err)
            {
                printf("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", TranslateOpenCLError(err));
                return 1;
            }

            platforms[i].devices[j].name = (cl_uchar*)malloc(stringLength * sizeof(cl_uchar));
            err = clGetDeviceInfo(devices_id[j], CL_DEVICE_NAME, stringLength, platforms[i].devices[j].name, NULL);
            if (CL_SUCCESS != err)
            {
                printf("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME returned '%s'.\n", TranslateOpenCLError(err));
                return 1;
            }

            // get device version
            err = clGetDeviceInfo(devices_id[j], CL_DEVICE_VERSION, 0, NULL, &stringLength);
            if (CL_SUCCESS != err)
            {
                printf("Error: clGetDeviceInfo() to get CL_PLATFORM_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
                return 1;
            }
            versionName = (char*)malloc(stringLength * sizeof(char));
            err = clGetDeviceInfo(devices_id[j], CL_DEVICE_VERSION, stringLength, versionName, NULL);
            if (CL_SUCCESS != err)
            {
                printf("Error: clGetDeviceInfo() to get CL_PLATFORM_NAME '%s'.\n", TranslateOpenCLError(err));
                return 1;
            }
            platforms[i].devices[j].version = getVersion(versionName);
            free(versionName);
        }
    }

    free(platforms_id);
    return 0;
}

cl_uint get_num_platforms()
{
    cl_uint numPlatforms = 0;
    cl_int err;

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (CL_SUCCESS != err)
    {
        printf("Error: clGetplatform_ids() to get num platforms returned %s.\n", TranslateOpenCLError(err));
        return 0;
    }
    return numPlatforms;
}

void PrintPlatforms(struct Platform* platforms, cl_uint num_platforms)
{
    for (cl_uint i = 0; i < num_platforms; ++i)
    {
        printf("Name of Platform: %s\n", platforms[i].name);

        for (cl_uint j = 0; j < platforms[i].num_devices; ++j)
        {
            printf("Name of Device: %s\n", platforms[i].devices[j].name);
        }
    }
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev) {

    cl_program program;
    char* program_log;
    size_t log_size;
    int err;

    /* Create program from file */
    program = clCreateProgramWithSource(ctx, 1,
        (const char**)&program_source, NULL, &err);
    if (err < 0) {
        printf("Couldn't create the program %s\n",TranslateOpenCLError(err));
        return NULL;
    }

    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        return NULL;
    }

    return program;
}

cl_float* genRandom(cl_ulong size)
{
    cl_float* data = (cl_float*)malloc(size * 2 * sizeof(cl_float));
    if (data == NULL)
    {
        printf("No Enougth memory\n");
        system("Pause");
    	return NULL;
    }

    srand(0);
    for (cl_ulong i = 0; i < size; i++) {
        data[2 * i] = (cl_float)rand();
        data[2 * i + 1] = (cl_float)rand();
    }

    return data;
}

#define INIT_FUNC "fft_init"
#define STAGE_FUNC "fft_stage"
#define SCALE_FUNC "fft_scale"

cl_int Cooley_Tukey(cl_float* data, cl_ulong const num_points, cl_int const direction, struct Device device) {

    /* Host/device data structures */
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel init_kernel, stage_kernel, scale_kernel;
    cl_int err;
    size_t global_size, local_size;
    cl_ulong local_mem_size;

    cl_ulong points_per_group, stage;
    cl_mem data_buffer;

    /* Create a context */
    context = clCreateContext(NULL, 1, &device.id, NULL, NULL, &err);
    if (err < 0) {
        printf("Couldn't create a context %s\n", TranslateOpenCLError(err));
        return 1;
    }

    /* Build the program */
    program = build_program(context, device.id);
	if(program == NULL)
	{
        printf("Couldn't create a program\n");

        clReleaseContext(context);
        return 1;
	}

    /* Create kernels for the FFT */
    init_kernel = clCreateKernel(program, INIT_FUNC, &err);
    if (err < 0) {
        printf("Couldn't create the initial kernel: %d", err);

        clReleaseContext(context);
        clReleaseProgram(program);
        return 1;
    };

    stage_kernel = clCreateKernel(program, STAGE_FUNC, &err);
    if (err < 0) {
        printf("Couldn't create the stage kernel: %s\n", TranslateOpenCLError(err));

        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseKernel(init_kernel);
        return 1;
    };

    scale_kernel = clCreateKernel(program, SCALE_FUNC, &err);
    if (err < 0) {
        printf("Couldn't create the scale kernel: %s\n", TranslateOpenCLError(err));

        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseKernel(init_kernel);
        clReleaseKernel(stage_kernel);
        return 1;
    }

    /* Create buffer */
    data_buffer = clCreateBuffer(context,
        CL_MEM_READ_WRITE,
        2 * num_points * sizeof(cl_float), NULL, &err);
    if (err < 0) {
        printf("Couldn't create a buffer: %s\n", TranslateOpenCLError(err));

        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseKernel(init_kernel);
        clReleaseKernel(stage_kernel);
        clReleaseKernel(scale_kernel);
        return 1;
    };

    /* Determine maximum work-group size */
    err = clGetKernelWorkGroupInfo(init_kernel, device.id,
        CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL);
    if (err < 0) {
        printf("Couldn't find the maximum work-group size: %s\n", TranslateOpenCLError(err));

        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseKernel(init_kernel);
        clReleaseKernel(stage_kernel);
        clReleaseKernel(scale_kernel);
        clReleaseMemObject(data_buffer);
        return 1;
    };
    local_size = local_size / 4;

    /* Determine local memory size */
    err = clGetDeviceInfo(device.id, CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof(local_mem_size), &local_mem_size, NULL);
    if (err < 0) {
        printf("Couldn't determine the local memory size: %s\n", TranslateOpenCLError(err));

        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseKernel(init_kernel);
        clReleaseKernel(stage_kernel);
        clReleaseKernel(scale_kernel);
        clReleaseMemObject(data_buffer);
        return 1;
    };

    /* Initialize kernel arguments */
    points_per_group = (cl_uint)(local_mem_size / (2 * sizeof(cl_float)));
    if (points_per_group > num_points)
        points_per_group = num_points;
	
    /* Set kernel arguments */
    err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &data_buffer);
    err |= clSetKernelArg(init_kernel, 1, (size_t)local_mem_size, NULL);
    err |= clSetKernelArg(init_kernel, 2, sizeof(points_per_group), &points_per_group);
    err |= clSetKernelArg(init_kernel, 3, sizeof(num_points), &num_points);
    err |= clSetKernelArg(init_kernel, 4, sizeof(direction), &direction);
    if (err < 0) {
        printf("Couldn't set a kernel argument");

        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseKernel(init_kernel);
        clReleaseKernel(stage_kernel);
        clReleaseKernel(scale_kernel);
        clReleaseMemObject(data_buffer);
        return 1;
    };

    /* Create a command queue */
    if (device.version >= 20) {
        const cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        queue = clCreateCommandQueueWithProperties(context, device.id, properties, &err);
    }
    else
    {
        queue = clCreateCommandQueue(context, device.id, CL_QUEUE_PROFILING_ENABLE, &err);
    }
    if (err < 0) {
        printf("Couldn't create a command queue: %s\n", TranslateOpenCLError(err));

        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseKernel(init_kernel);
        clReleaseKernel(stage_kernel);
        clReleaseKernel(scale_kernel);
        clReleaseMemObject(data_buffer);
        return 1;
    };
	
    /* Enqueue data write */
    err = clEnqueueWriteBuffer(queue, data_buffer, CL_FALSE, 0,
        2 * num_points * sizeof(cl_float), data, 0, NULL, NULL);
    if (err < 0) {
        printf("Couldn't write tu buffer: %s\n", TranslateOpenCLError(err));

        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseKernel(init_kernel);
        clReleaseKernel(stage_kernel);
        clReleaseKernel(scale_kernel);
        clReleaseMemObject(data_buffer);
        clRetainCommandQueue(queue);
        return 1;
    }

    if (local_size > num_points/4)
        local_size = num_points/4;

    /* Enqueue initial kernel */
    global_size = (num_points / points_per_group) * local_size;
    //printf("global_size: %u, currently %d\n", global_size, num_points);
    //printf("CL_KERNEL_WORK_GROUP_SIZE: %u\n", local_size);
    //fflush(stdout);
    err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &global_size,
        &local_size, 0, NULL, NULL);
    if (err < 0) {
        printf("Couldn't enqueue the initial kernel: %s\n", TranslateOpenCLError(err));

        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseKernel(init_kernel);
        clReleaseKernel(stage_kernel);
        clReleaseKernel(scale_kernel);
        clReleaseMemObject(data_buffer);
        clRetainCommandQueue(queue);
        return 1;
    }

    /* Enqueue further stages of the FFT */
    if (num_points > points_per_group) {

        err = clSetKernelArg(stage_kernel, 0, sizeof(cl_mem), &data_buffer);
        err |= clSetKernelArg(stage_kernel, 2, sizeof(points_per_group), &points_per_group);
        err |= clSetKernelArg(stage_kernel, 3, sizeof(direction), &direction);
        if (err < 0) {
            printf("Couldn't set a kernel argument: %s\n", TranslateOpenCLError(err));

            clReleaseContext(context);
            clReleaseProgram(program);
            clReleaseKernel(init_kernel);
            clReleaseKernel(stage_kernel);
            clReleaseKernel(scale_kernel);
            clReleaseMemObject(data_buffer);
            clRetainCommandQueue(queue);
            return 1;
        };
        for (stage = 2; stage <= num_points / points_per_group; stage <<= 1) {
            clSetKernelArg(stage_kernel, 1, sizeof(stage), &stage);
            err = clEnqueueNDRangeKernel(queue, stage_kernel, 1, NULL, &global_size,
                &local_size, 0, NULL, NULL);
            if (err < 0) {
                printf("Couldn't enqueue the stage kernel: %s\n", TranslateOpenCLError(err));

                clReleaseContext(context);
                clReleaseProgram(program);
                clReleaseKernel(init_kernel);
                clReleaseKernel(stage_kernel);
                clReleaseKernel(scale_kernel);
                clReleaseMemObject(data_buffer);
                clRetainCommandQueue(queue);
                return 1;
            }
        }
    }

    /* Scale values if performing the inverse FFT */
    if (direction < 0) {
        err = clSetKernelArg(scale_kernel, 0, sizeof(cl_mem), &data_buffer);
        err |= clSetKernelArg(scale_kernel, 1, sizeof(points_per_group), &points_per_group);
        err |= clSetKernelArg(scale_kernel, 2, sizeof(num_points), &num_points);
        if (err < 0) {
            printf("Couldn't set a kernel argument: %s\n", TranslateOpenCLError(err));

            clReleaseContext(context);
            clReleaseProgram(program);
            clReleaseKernel(init_kernel);
            clReleaseKernel(stage_kernel);
            clReleaseKernel(scale_kernel);
            clReleaseMemObject(data_buffer);
            clRetainCommandQueue(queue);
            return 1;
        };
        err = clEnqueueNDRangeKernel(queue, scale_kernel, 1, NULL, &global_size,
            &local_size, 0, NULL, NULL);
        if (err < 0) {
            printf("Couldn't enqueue the initial kernel: %s\n", TranslateOpenCLError(err));

            clReleaseContext(context);
            clReleaseProgram(program);
            clReleaseKernel(init_kernel);
            clReleaseKernel(stage_kernel);
            clReleaseKernel(scale_kernel);
            clReleaseMemObject(data_buffer);
            clRetainCommandQueue(queue);
            return 1;
        }
    }

    clFinish(queue);
    /* Read the results */
    err = clEnqueueReadBuffer(queue, data_buffer, CL_TRUE, 0,
        num_points * 2 * sizeof(cl_float), data, 0, NULL, NULL);
    if (err < 0) {
        printf("Couldn't read the buffer: %s\n", TranslateOpenCLError(err));

        clReleaseContext(context);
        clReleaseProgram(program);
        clReleaseKernel(init_kernel);
        clReleaseKernel(stage_kernel);
        clReleaseKernel(scale_kernel);
        clReleaseMemObject(data_buffer);
        clRetainCommandQueue(queue);
        return 1;
    }

    /* Deallocate resources */
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(init_kernel);
    clReleaseKernel(stage_kernel);
    clReleaseKernel(scale_kernel);
    clReleaseMemObject(data_buffer);
    clRetainCommandQueue(queue);

    return 0;
}

void free_platforms(struct Platform* platforms, cl_uint num_platforms)
{
    for (cl_uint i = 0; i < num_platforms; ++i)
    {
        free(platforms[i].name);
    	
        for (cl_uint j = 0; j < platforms[i].num_devices; ++j)
        {
            free(platforms[i].devices[j].name);
            clReleaseDevice(platforms[i].devices[j].id);
		}
    	
        free(platforms[i].devices);
	}
    free(platforms);
}

#define NUM_POINTS 4096//16777216//33554432//67108864//134217728//////4294967296//1048576//262144//16384//8192 //4096 //16384 //262144 //4096

int main(int argc, char* argv[])
{
    cl_uint num_platforms = get_num_platforms();
    printf("Numbers of platforms: %d\n", num_platforms);
    struct Platform* platforms = (struct Platform*)malloc(num_platforms * sizeof(struct Platform));
    get_platforms(platforms, num_platforms);
    PrintPlatforms(platforms, num_platforms);


    // Algo
    for (cl_uint i = 0; i < num_platforms; ++i)
    {
        for (cl_uint j = 0; j < platforms[i].num_devices; ++j)
        {
            cl_float* data = genRandom(NUM_POINTS);
        	
            if (Cooley_Tukey(data, NUM_POINTS, 1, platforms[i].devices[j]) == 0)
                printf("Succes\n");
            else
                printf("Fail\n");

			free(data);
        }
    }

    free_platforms(platforms, num_platforms);
    system("Pause");
    return 0;
}