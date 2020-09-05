using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace FFTGUI.MVVMbasics
{
    abstract class ViewModel : INotifyPropertyChanged, IDataErrorInfo
    {
        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public abstract string this[string propertyName] { get; }
        public abstract string Error { get; }
    }
}
