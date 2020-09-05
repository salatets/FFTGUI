using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Drawing;

namespace FFTGUI
{
    class FFTModel
    {
        public struct DeviceData
        {
            public string Name { get; set; }
            public ApiWrapper.CLDeviceID Id;
            public uint Version { get; set; }
        }

        public struct PlatformData
        {
            public string Name { get; set; }
            public uint Version { get; set; }
            public uint NumDevices { get; set; }
            public DeviceData[] Devices { get; set; }

            public ApiWrapper.Device[] devices;
        }

        private float[] _inputData = null;
        public float[] InputData => _inputData;

        private float[] _outputData = null;
        public float[] OutputData => _outputData;

        public static PlatformData[] GetPlatforms()
        {
            uint platformCount = ApiWrapper.get_num_platforms();

            if (platformCount == 0)
                return null;

            ApiWrapper.Platform[] data = new ApiWrapper.Platform[platformCount];
            PlatformData[] retData = new PlatformData[platformCount];

            unsafe
            {
                fixed (ApiWrapper.Platform* pData = data)
                {
                    ApiWrapper.get_platforms(pData, platformCount);
                }

                for (uint i = 0; i < platformCount; ++i)
                {
                    retData[i].Name = Marshal.PtrToStringAnsi((IntPtr)data[i].name);
                    retData[i].NumDevices = data[i].num_devices;
                    retData[i].Devices = new DeviceData[retData[i].NumDevices];
                    retData[i].devices = new ApiWrapper.Device[retData[i].NumDevices];

                    for (uint j = 0; j < retData[i].NumDevices; ++j)
                    {
                        retData[i].devices[j] = data[i].devices[j];

                        retData[i].Devices[j] = new DeviceData
                        {
                            Id = data[i].devices[j].id,
                            Name = Marshal.PtrToStringAnsi((IntPtr) data[i].devices[j].name),
                            Version = data[i].devices[j].version
                        };
                    }
                }
            }
            //TODO add free
            return retData;
        }

        public bool Cooley_Tukey(ApiWrapper.Device device, bool isInverse = false)
        {
            int status;
            var data = new float[_inputData.Length];
            _inputData.CopyTo(data, 0);
            unsafe
            {

                fixed (float* pData = data)
                {
                    status = ApiWrapper.Cooley_Tukey(pData, (uint)data.Length/2, isInverse ? -1 : 1, device);
                }
            }

            if (status == 0)
                _outputData = data;

            return status == 0;
        }

        public bool ReadData(string path)
        {
            float[] data;

            if (path.Substring(path.Length - 5) == ".data")
            {
                TextReader reader = File.OpenText(path);
                data = ParseInput(reader);
            }
            else
            {
                try
                {
                    var image = new Bitmap(path, true);
                    data = ParseImageLine(image);
                }
                catch (Exception)
                {
                    return false;
                }
                
            }
            var isValid = ValidateInput(ref data);
            if (isValid)
                _inputData = data;
            return isValid;
        }

        // from Lightness
        private static float[] ParseImageLine(Bitmap image)
        {
            float[] ret = new float[image.Width * 2];

            for (int i = 0; i < image.Width; ++i)
            {
                ret[2 * i + 1] = 0;
                ret[2 * i] = image.GetPixel(i,0).GetBrightness();
            }

            return ret;
        }

        // Assumption background is white
        private static float[] ParseGraphicImage(Bitmap image)
        {
            float[] ret = new float[image.Width * 2];

            for (int i = 0; i < image.Width; ++i)
            {
                ret[2 * i + 1] = 0;
                for (int j = 0; j < image.Height; ++j)
                {
                    if (image.GetPixel(i, j).GetBrightness() < Color.DarkGray.GetBrightness())
                    {
                        ret[2*i] = j;
                    }
                }
            }

            return ret;
        }

        private static float[] ParseInput(TextReader reader)
        {
            List<float> data = new List<float>();
            while (true)
            {
                String[] strs = reader.ReadLine()?.Split(' ');
                if (strs == null || strs.Length == 0)
                    break;
                foreach (var str in strs)
                {
                    float temp;
                    if (float.TryParse(str, out temp))
                        data.Add(temp);
                }
            }

            return data.ToArray();
        }

        /* check if size of data is power of two */
        private static bool ValidateInput(ref float[] data)
        {
            if (data.Length % 2 == 1)
                return false;

            uint x = (uint)data.Length / 2; // we work with complex numbers
            return (x & (x - 1)) == 0;
        }

        public void GenData(uint num_points)
        {
            float[] data = new float[num_points * 2];

            var rand = new Random();

            switch (rand.Next(0, 6))
            {
                case 0:
                    for (uint i = 0; i < num_points; i++)
                    {
                        data[2 * i] = (float)Math.Sin(i / (num_points / 64 +1)) * (float)rand.NextDouble() * 5 +
                                      (float)rand.NextDouble();
                        data[2 * i + 1] = (float)Math.Sin(i / (num_points / 64 +1) + Math.PI / 2) * (float)rand.NextDouble() * 2 +
                                          (float)rand.NextDouble();
                    }
                    break;
                case 1:
                    for (uint i = 0; i < num_points; i++)
                    {
                        data[2 * i] = (float)Math.Sin(i/num_points) * (float)rand.NextDouble() * 10 +
                                      (float)Math.Sin(i / (num_points/16 +1)) * (float)rand.NextDouble() * 5 +
                                      (float)Math.Sin(i / (num_points / 256 +1)) * (float)rand.NextDouble() * 2 +
                                      (float)rand.NextDouble() * 2;
                        data[2 * i + 1] = (float)Math.Sin(i / (num_points/8 +1)) * (float)rand.NextDouble() * 10 +
                                          (float)Math.Sin(i / (num_points / 256 +1)) * (float)rand.NextDouble() * 5 +
                                          (float)Math.Sin(i / (num_points / 64 +1)) * (float)rand.NextDouble() * 2 + 
                                          (float)rand.NextDouble() * 2;
                    }
                    break;
                case 2:
                    for (uint i = 0; i < num_points; i++)
                    {
                        data[2 * i] = (float) Math.Sin(Math.Pow(i,2) / (num_points / 256 +1)) * 5 +
                                      (float) rand.NextDouble() * 2;
                        data[2 * i + 1] = (float)Math.Cos(Math.Pow(i, 2) / (num_points / 256 + 1)) * 5 +
                                          (float)rand.NextDouble() * 2;
                    }
                    break;
                case 3:
                    for (uint i = 0; i < num_points; i++)
                    {
                        data[2 * i] = (float)Math.Sin(Math.Cos(5/2 * i/(num_points / 128 + 1))) * 10 +
                                      (float)rand.NextDouble() * 2;
                        data[2 * i + 1] = (float)Math.Cos(Math.Sin(2/5 * i / (num_points / 128 + 1))) * 6 +
                                          (float)rand.NextDouble() * 2;
                    }
                    break;
                case 4:
                    for (uint i = 0; i < num_points; i++)
                    {
                        data[2 * i] = (float)Math.Sin(num_points/(i+1)) * 5 +
                                      (float)rand.NextDouble() * 2;
                        data[2 * i + 1] = (float)Math.Cos(num_points /(i + 1)) * 5 +
                                          (float)rand.NextDouble() * 2;
                    }
                    break;
                case 5:
                    for (uint i = 0; i < num_points; i++)
                    {
                        data[2 * i] = (float) Math.Sin(Math.PI * ((float)i / (num_points + 1) * 8)) * 10;
                        data[2 * i + 1] = (float)Math.Cos(Math.PI * ((float)i / (num_points + 1) * 8)) * 10;
                    }

                    break;
            }

            _inputData = data;
        }

        public void ClearData(bool isInput)
        {
            if (isInput)
                _inputData = null;
            else
                _outputData = null;
        }
    }
}
