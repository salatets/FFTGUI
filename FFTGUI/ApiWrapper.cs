using System;
using System.Runtime.InteropServices;

namespace FFTGUI
{
    public static class ApiWrapper
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct CLDeviceID
        {
            public IntPtr Value;
        }

        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct Device
        {
            public byte* name;
            public CLDeviceID id;
            public uint version;
        }

        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct Platform
        {
            public byte* name;
            uint version;
            public uint num_devices;
            public Device* devices;
        }

        [DllImport("FFT.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern unsafe int get_platforms(Platform* platforms, uint numPlatforms);

        [DllImport("FFT.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint get_num_platforms();

        [DllImport("FFT.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int Cooley_Tukey(float* data, ulong num_points, int direction, Device device);

        [DllImport("FFT.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe void free_platforms(Platform* platforms, uint num_platforms);
    }
}
