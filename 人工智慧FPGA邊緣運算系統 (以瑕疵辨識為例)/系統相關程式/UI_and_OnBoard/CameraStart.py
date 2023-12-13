from PyQt5 import QtWidgets

from CameraController import MainWindow_controller

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller("/home/petalinux/SN_Quantization.xmodel")
    window.show()
    sys.exit(app.exec_())
