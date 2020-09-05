using System;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Input;
using FFTGUI.MVVMbasics;
using Microsoft.Win32;
using OxyPlot;
using OxyPlot.Series;

namespace FFTGUI
{
    //TODO May be remade radiobutton bindings

    class FftViewModel : ViewModel
    {
        
        private static FFTModel Model = new FFTModel();

        /* View Properties */

        public PlotModel InputGraph { get; } = new PlotModel { Title = "input data" };
        public PlotModel OutputGraph { get; } = new PlotModel { Title = "output data" };

        private FFTModel.PlatformData[] _platformData = FFTModel.GetPlatforms();
        public FFTModel.PlatformData[] PlatformData
        {
            get
            {
                if (_platformData == null)
                {
                    MessageBox.Show("OpenCL platform not found", "FFTGUI", MessageBoxButton.OK, MessageBoxImage.Stop);
                    System.Windows.Application.Current.Shutdown();
                }

                return _platformData;
            }
        } 

        private FFTModel.PlatformData _platform;
        public FFTModel.PlatformData Platform
        {
            get => _platform;
            set
            {
                _platform = value;
                OnPropertyChanged();
            }
        }

        public uint DeviceId { get; set; } = 0;

        private uint _pointsGenerated = 512;
        public uint PointsGenerated
        {
            get => _pointsGenerated;
            set
            {
                if (_pointsGenerated == value)
                    return;

                _pointsGenerated = value;
                OnPropertyChanged();
            }
        }
        
        private enum DataSource
        {
            Random,
            File
        }

        private DataSource _sourceOfData = DataSource.Random;
        private DataSource SourceOfData
        {
            get => _sourceOfData;
            set
            {
                if (_sourceOfData == value)
                    return;

                if (value == DataSource.Random)
                {
                    IsEnableRandomSource = true;
                    IsEnableFileSource = false;
                    
                }
                else
                {
                    IsEnableFileSource = true;
                    IsEnableRandomSource = false;
                }

                _sourceOfData = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(IsRandomSource));
                OnPropertyChanged(nameof(IsFileSource));

                OnPropertyChanged(nameof(IsEnableFileSource));
                OnPropertyChanged(nameof(IsEnableRandomSource));
            }
        }

        public bool IsRandomSource
        {
            get => SourceOfData == DataSource.Random;
            set => SourceOfData = value ? DataSource.Random : SourceOfData;
        }
        public bool IsFileSource
        {
            get => SourceOfData == DataSource.File;
            set => SourceOfData = value ? DataSource.File : SourceOfData;
        }

        public bool IsEnableRandomSource { get; set; } = true;
        public bool IsEnableFileSource { get; set; }

        public bool IsInverse { get; set; }

        private enum GraphicType
        {
            Аmplitude,
            Phase,
            TwoComponent
        }

        private GraphicType _typeOfGraphic = GraphicType.Аmplitude;
        private GraphicType TypeOfGraphic
        {
            get => _typeOfGraphic;
            set
            {
                if (_typeOfGraphic == value)
                    return;

                _typeOfGraphic = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(IsAmplitudeGraphic));
                OnPropertyChanged(nameof(IsTwoComponentGraphic));
                OnPropertyChanged(nameof(IsPhaseGraphic));

                OnGraphicChange(isInput: true);
                OnGraphicChange(isInput: false);
            }
        }

        public bool IsAmplitudeGraphic
        {
            get => TypeOfGraphic == GraphicType.Аmplitude;
            set => TypeOfGraphic = value ? GraphicType.Аmplitude : TypeOfGraphic;
        }
        public bool IsPhaseGraphic
        {
            get => TypeOfGraphic == GraphicType.Phase;
            set => TypeOfGraphic = value ? GraphicType.Phase : TypeOfGraphic;
        }
        public bool IsTwoComponentGraphic
        {
            get => TypeOfGraphic == GraphicType.TwoComponent;
            set => TypeOfGraphic = value ? GraphicType.TwoComponent : TypeOfGraphic;
        }


        /* Commands */

        private RelayCommand _generateCommand;

        public ICommand GenerateCommand
        {
            get
            {
                return _generateCommand ?? (_generateCommand = // cause non static function
                    new RelayCommand(
                        param => this.GenData(), 
                        param => this.CanGenerate
                        ));
            }
        }

        private RelayCommand _convertCommand;
        public ICommand ConvertCommand
        {
            get
            {
                return _convertCommand ?? (_convertCommand =
                    new RelayCommand(
                        param => this.ConvertData(),
                        param => this.CanConvert
                    ));
            }
        }

        private RelayCommand _readFileCommand;
        public ICommand ReadFileCommand
        {
            get
            {
                return _readFileCommand ?? (_readFileCommand =
                    new RelayCommand(
                        param => this.ReadFile(),
                        param => true
                    ));
            }
        }

        private RelayCommand _writeFileCommand;

        public ICommand WriteFileCommand
        {
            get
            {
                return _writeFileCommand ?? (_writeFileCommand =
                    new RelayCommand(
                        param => this.WriteFile(),
                        param => this.CanWrite
                    ));
            }
        }


        /* Command's Executors */

        private void GenData()
        {
            Model.GenData(_pointsGenerated);
            OnGraphicChange(isInput: true);
            InvalidateGraphic(false);
        }

        private void ReadFile()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            openFileDialog.Filter = "Data files (*.data)|*.data|All files (*.*)|*.*";


            if (openFileDialog.ShowDialog() == true)
            {
                if (Model.ReadData(openFileDialog.FileName))
                {
                    OnGraphicChange(isInput: true);
                    InvalidateGraphic(false);
                }
                else
                {
                    MessageBox.Show(
                        "Cannot read data, maybe no appropriate formal\n" +
                        " or file content length not power of 2",
                        "FFTGUI", MessageBoxButton.OK, MessageBoxImage.Stop);
                }
            }
        }

        private void ConvertData()
        {
            if (Platform.NumDevices <= DeviceId)
                return;

            if (Model.Cooley_Tukey(Platform.devices[DeviceId], IsInverse))
                OnGraphicChange(isInput: false);
        }

        private void WriteFile()
        {
            SaveFileDialog saveFileDialog = new SaveFileDialog();
            saveFileDialog.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            saveFileDialog.Filter = "Data files (*.data)|*.data|All files (*.*)|*.*";

            if (saveFileDialog.ShowDialog() == true)
                File.WriteAllText(saveFileDialog.FileName, DataToString());
        }


        /* Command's CanExecutors */

        private bool CanGenerate
        {
            get => ValidateGeneratedNum() == null;
        }

        private bool CanConvert
        {
            get => ValidateInputData() == null;
        }

        private bool CanWrite
        {
            get => ValidateOutputData() == null;
        }


        /* Input Validators */

        private string ValidateOutputData()
        {
            if (Model.OutputData != null)
                return null;

            return "Nothing to write";
        }

        private string ValidateGeneratedNum()
        {
            /* check if number is power of two */
            if ((_pointsGenerated & (_pointsGenerated - 1)) == 0)
                return null;

            return "Number must be power of two";
        }

        private string ValidateInputData()
        {
            if (Model.InputData != null)
                return null;

            return "Data is not selected";
        }


        /* Graphic draw helpers functions */

        private void OnGraphicChange(bool isInput)
        {
            PlotModel plot;
            float[] data;

            if (isInput)
            {
                plot = InputGraph;
                data = Model.InputData;
            }
            else
            {
                plot = OutputGraph;
                data = Model.OutputData;
            }

            if (data == null)
                return;

            FunctionSeries[] series;

            if (IsAmplitudeGraphic)
                series = new FunctionSeries[]
                {
                    new FunctionSeries((d => GetData_Module(ref data, d)), 0, data.Length / 2 - 1, 1.0) {Color = OxyColors.MediumSeaGreen}
                };
            else if(IsTwoComponentGraphic)
                series = new FunctionSeries[]
                {
                    new FunctionSeries((d => GetData_Real(ref data, d)), 0, data.Length / 2 - 1, 1.0) {Color = OxyColors.OrangeRed},
                    new FunctionSeries((d => GetData_Img(ref data, d)), 0, data.Length / 2 - 1, 1.0) {Color = OxyColors.DeepSkyBlue}
                };
            else
                series = new FunctionSeries[]
                {
                    new FunctionSeries((d => GetData_Angle(ref data, d)), 0, data.Length / 2 - 1, 1.0) {Color = OxyColors.MediumPurple}
                };

            plot.Series.Clear();
            foreach (var part in series)
            {
                plot.Series.Add(part);
            }
            plot.InvalidatePlot(true);
        }

        void InvalidateGraphic(bool isInput)
        {
            PlotModel plot;
            if (isInput)
            {
                plot = InputGraph;
            }
            else
            {
                plot = OutputGraph;
            }

            plot.Series.Clear();
            plot.InvalidatePlot(true);
            Model.ClearData(isInput);
        }

        

        private double GetData_Real(ref float[] data, double index)
        {
            return data[2 * (int)index];
        }

        private double GetData_Img(ref float[] data, double index)
        {
            return data[2 * (int)index + 1];
        }

        private double GetData_Module(ref float[] data, double index)
        {
            return Math.Sqrt(
                Math.Pow(data[2 * (int)index + 1], 2) +
                Math.Pow(data[2 * (int)index], 2)
            );
        }

        private double GetData_Angle(ref float[] data, double index)
        {
            return Math.Atan2(
                data[2 * (int) index + 1],
                data[2 * (int)index]
            );
        }


        /* Utilities */
        private string DataToString()
        {
            StringBuilder sb = new StringBuilder(Model.OutputData.Length * 2);
            foreach (var num in Model.OutputData)
            {
                sb.Append(num);
                sb.Append(" ");
            }

            return sb.ToString();
        }

        public override string this[string propertyName]
        {
            get
            {
                string error = null;

                if (propertyName == "PointsGenerated")
                    error = ValidateGeneratedNum();

                return error;
            }
        }

        public override string Error { get; }

    }
}
