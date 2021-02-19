import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.lbl = QLabel(self)
        self.lbl.move(60, 40)

        qle = QLineEdit(self)
        qle.move(60, 100)
        btn1 = QPushButton('&Button1', self)
        btn1.setCheckable(True)
        btn1.clicked.connect(self.btn_clicked())

        qle.textChanged[str].connect(self.onChanged)

        vbox = QVBoxLayout()
        vbox.addWidget(btn1)

        self.setLayout(vbox)
        self.resize(800, 400)
        self.setWindowTitle('QLineEdit')
        self.show()

    def onChanged(self, text):
        self.lbl.setText(text)
        self.lbl.adjustSize()

    def btn_clicked(self):
        QmessageBox.about(self,"message","clicked")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())