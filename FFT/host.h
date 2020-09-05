#pragma once
#include <CL/cl.h>

struct Device
{
    cl_uchar* name;
    cl_device_id id;
    cl_uint version;
};

struct Platform
{
    cl_uchar* name;
    cl_uint version;
    cl_uint num_devices;
    struct Device* devices;
};

__declspec(dllexport) cl_int get_platforms(struct Platform* platforms, cl_uint num_platforms);

__declspec(dllexport) cl_uint get_num_platforms();

__declspec(dllexport) cl_int Cooley_Tukey(cl_float* data, cl_ulong const num_points, cl_int const direction, struct Device device);

__declspec(dllexport) void free_platforms(struct Platform* platforms, cl_uint num_platforms);