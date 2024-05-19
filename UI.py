from PyQt5 import QtCore, QtGui, QtWidgets
from PipeLine import PipeLine
from config_file import load_config, update_config, save_config
from library.custom_logging import setup_logging
from library.StatObjectDecider import StatObjectDecider
import os
import sys
import subprocess


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.directories = _get_dir_name_()
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 1000)
        MainWindow.setStyleSheet("background-color:rgb(191, 224, 203)\n"
                                 "")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Метка Первый Русский поисковик парнокопытных!
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(250, 50, 980, 50))
        self.label_1.setStyleSheet("font-size:48px;\n"
                                   "font: 75 \"Comic Sans MS\";\n"
                                   "color:rgb(67, 74, 84)\n"
                                   "")
        self.label_1.setObjectName("label_1")

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 110, 525, 420))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.checkBox_2 = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox_2.setStyleSheet("font-size:12px;\n"
                                      "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                      "color:rgb(67, 74, 84);\n"
                                      "")
        self.checkBox_2.setObjectName("checkBox_2")
        self.verticalLayout.addWidget(self.checkBox_2)

        self.checkBox_3 = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox_3.setStyleSheet("font-size:12px;\n"
                                      "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                      "color:rgb(67, 74, 84);\n"
                                      "")
        self.checkBox_3.setObjectName("checkBox_3")
        self.verticalLayout.addWidget(self.checkBox_3)

        self.checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox.setStyleSheet("font-size:12px;\n"
                                    "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                    "color:rgb(67, 74, 84);\n"
                                    "")
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout.addWidget(self.checkBox)

        # Выпадающий список
        self.comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.comboBox.setStyleSheet("background-color: #31A05E;\n"
                                    "font-size:24px;\n"
                                    "font: 75 \"Comic Sans MS\";\n"
                                    "color:rgb(249, 246, 229);\n"
                                    "")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Круг")
        self.comboBox.addItem("Квадрат")
        self.comboBox.setCurrentIndex(0)
        self.verticalLayout.addWidget(self.comboBox)

        # Создание кнопки
        self.push_button_13_3 = QtWidgets.QPushButton("Сохранить настройки")
        self.push_button_13_3.setStyleSheet("background-color: #31A05E;\n"
                                            "font-size:24px;\n"
                                            "font: 75 \"Comic Sans MS\";\n"
                                            "color:rgb(249, 246, 229);\n"
                                            "")
        self.verticalLayout.addWidget(self.push_button_13_3)

        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 240, 131, 380))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.horizontalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.spinBox = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.spinBox.setStyleSheet("font-size:12px;\n"
                                   "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                   "color:rgb(67, 74, 84);\n"
                                   "")
        self.spinBox.setObjectName("spinBox")
        self.verticalLayout_2.addWidget(self.spinBox)
        self.spinBox.setValue(5)
        self.spinBox.setRange(1, 1000)

        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.horizontalLayoutWidget)
        self.doubleSpinBox_2.setStyleSheet("font-size:12px;\n"
                                           "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                           "color:rgb(67, 74, 84);\n"
                                           "")
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.verticalLayout_2.addWidget(self.doubleSpinBox_2)
        self.doubleSpinBox_2.setValue(0.5)
        self.doubleSpinBox_2.setRange(0.01, 1.0)
        self.doubleSpinBox_2.setSingleStep(0.05)

        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.horizontalLayoutWidget)
        self.doubleSpinBox_3.setStyleSheet("font-size:12px;\n"
                                           "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                           "color:rgb(67, 74, 84);\n"
                                           "")
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.verticalLayout_2.addWidget(self.doubleSpinBox_3)
        self.doubleSpinBox_3.setValue(0.95)
        self.doubleSpinBox_3.setRange(0.01, 1.0)
        self.doubleSpinBox_3.setSingleStep(0.05)

        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.horizontalLayoutWidget)
        self.doubleSpinBox.setStyleSheet("font-size:12px;\n"
                                         "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                         "color:rgb(67, 74, 84);\n"
                                         "")
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.verticalLayout_2.addWidget(self.doubleSpinBox)
        self.doubleSpinBox.setValue(0.5)
        self.doubleSpinBox.setRange(0.01, 1.0)
        self.doubleSpinBox.setSingleStep(0.05)

        self.doubleSpinBox_4 = QtWidgets.QDoubleSpinBox(self.horizontalLayoutWidget)
        self.doubleSpinBox_4.setStyleSheet("font-size:12px;\n"
                                           "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                           "color:rgb(67, 74, 84);\n"
                                           "")
        self.doubleSpinBox_4.setObjectName("doubleSpinBox_4")
        self.verticalLayout_2.addWidget(self.doubleSpinBox_4)
        self.doubleSpinBox_4.setValue(0)
        self.doubleSpinBox_4.setRange(0, 1000)

        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(10, 110, 330, 350))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout")

        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.label_2.setStyleSheet("font-size:12px;\n"
                                   "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                   "color:rgb(67, 74, 84);\n"
                                   "")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)

        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.label_3.setStyleSheet("font-size:12px;\n"
                                   "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                   "color:rgb(67, 74, 84);\n"
                                   "")
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)

        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.label_4.setStyleSheet("font-size:12px;\n"
                                   "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                   "color:rgb(67, 74, 84);\n"
                                   "")
        self.label_4.setObjectName("label_4")
        self.verticalLayout_4.addWidget(self.label_4)

        self.label_5 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.label_5.setStyleSheet("font-size:12px;\n"
                                   "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                   "color:rgb(67, 74, 84);\n"
                                   "")
        self.label_5.setObjectName("label_5")
        self.verticalLayout_4.addWidget(self.label_5)

        self.label_8 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.label_8.setStyleSheet("font-size:12px;\n"
                                   "font: 75 italic 12pt \"Comic Sans MS\";\n"
                                   "color:rgb(67, 74, 84);\n"
                                   "")
        self.label_8.setObjectName("label_8")
        self.verticalLayout_4.addWidget(self.label_8)

        # Установка рамки основного меню
        self.frame = QtWidgets.QFrame(self.centralwidget)
        main_window_width = MainWindow.width()
        main_window_height = MainWindow.height()
        frame_width = 600
        frame_height = 300
        frame_x = (main_window_width - frame_width) / 2 + 320  # Центрируем по горизонтали
        frame_y = main_window_height - frame_height  # Размещаем внизу
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(frame_x, frame_y - 50, frame_width, frame_height))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)  # Форма рамки
        self.frame.setLineWidth(2)  # Ширина рамки

        # Создание вертикального макета для рамки
        self.frame_layout = QtWidgets.QVBoxLayout(self.frame)
        self.frame_layout.setContentsMargins(20, 20, 20, 20)  # Устанавливаем отступы

        self.lineEdit = QtWidgets.QLineEdit(self.frame)
        self.lineEdit.setStyleSheet("font-size:24px;\n"
                                    "font: 75 italic 8pt \"Comic Sans MS\";\n"
                                    "color:rgb(67, 74, 84);\n"
                                    "")
        self.lineEdit.setObjectName("lineEdit")
        self.frame_layout.addWidget(self.lineEdit)
        self.lineEdit.setAlignment(QtCore.Qt.AlignCenter)

        # Создание надписи и добавление ее в макет
        self.label_6 = QtWidgets.QLabel("Укажите директорию к папке (только изображения или только видео)")
        self.label_6.setStyleSheet("font-size:24px;\n"
                                   "font: 75 \"Comic Sans MS\";\n"
                                   "color:rgb(67, 74, 84);\n"
                                   "")
        self.frame_layout.addWidget(self.label_6)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)

        self.lineEdit.setAlignment(QtCore.Qt.AlignCenter)

        # Создание кнопки
        self.pushButton_13 = QtWidgets.QPushButton("Запустить вычисления!")
        self.pushButton_13.setStyleSheet("background-color: #31A05E;\n"
                                         "font-size:24px;\n"
                                         "font: 75 \"Comic Sans MS\";\n"
                                         "color:rgb(249, 246, 229);\n"
                                         "")

        # Создание кнопки
        self.pushButton_12 = QtWidgets.QPushButton("Указать директорию")
        self.pushButton_12.setStyleSheet("background-color: #31A05E;\n"
                                         "font-size:24px;\n"
                                         "font: 75 \"Comic Sans MS\";\n"
                                         "color:rgb(249, 246, 229);\n"
                                         "")

        # Установка фиксированного размера для кнопки
        self.pushButton_13.setFixedSize(QtCore.QSize(frame_width - 180, 48))
        self.pushButton_12.setFixedSize(QtCore.QSize(frame_width - 180, 48))
        self.label_6.setFixedSize(QtCore.QSize(frame_width - 40, 48))
        self.lineEdit.setFixedSize(QtCore.QSize(frame_width - 180, 48))

        # Создание растягивающегося элемента для центрирования кнопки по вертикали
        self.frame_layout.addStretch(1)

        # Добавление кнопки в макет
        self.frame_layout.addWidget(self.pushButton_13)
        self.frame_layout.addWidget(self.pushButton_12)
        self.frame_layout.addWidget(self.lineEdit)

        # Установка макета для рамки
        self.frame.setLayout(self.frame_layout)

        self.button_layout_2 = QtWidgets.QHBoxLayout()
        self.button_layout_2.addStretch(1)  # Добавляем растягивающийся элемент для центрирования
        self.button_layout_2.addWidget(self.pushButton_13)  # Добавляем кнопку
        self.button_layout_2.addStretch(1)  # Добавляем еще один растягивающийся элемент для центрирования
        self.frame_layout.addLayout(self.button_layout_2)

        self.button_layout_1 = QtWidgets.QHBoxLayout()
        self.button_layout_1.addStretch(1)  # Добавляем растягивающийся элемент для центрирования
        self.button_layout_1.addWidget(self.lineEdit)  # Добавляем кнопку
        self.button_layout_1.addStretch(1)  # Добавляем еще один растягивающийся элемент для центрирования
        self.frame_layout.addLayout(self.button_layout_1)

        # Создание горизонтального макета для кнопки и центрирование
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addStretch(1)  # Добавляем растягивающийся элемент для центрирования
        self.button_layout.addWidget(self.pushButton_12)  # Добавляем кнопку
        self.button_layout.addStretch(1)  # Добавляем еще один растягивающийся элемент для центрирования

        # Добавляем горизонтальный макет с кнопкой в вертикальный макет рамки
        self.frame_layout.addLayout(self.button_layout)

        frame_2 = QtWidgets.QFrame(self.centralwidget)
        main_window_width = MainWindow.width()
        main_window_height = MainWindow.height()
        frame_width = 600
        frame_height = 300
        frame_x = (main_window_width - frame_width) / 2 - 320  # Центрируем по горизонтали
        frame_y = main_window_height - frame_height  # Размещаем внизу
        frame_2 = QtWidgets.QFrame(self.centralwidget)
        frame_2.setGeometry(QtCore.QRect(frame_x, frame_y - 50, frame_width, frame_height))
        frame_2.setFrameShape(QtWidgets.QFrame.Box)  # Форма рамки
        frame_2.setLineWidth(2)  # Ширина рамки

        # Создание вертикального макета для рамки
        frame_layout_2 = QtWidgets.QVBoxLayout(frame_2)
        frame_layout_2.setContentsMargins(20, 20, 20, 20)  # Устанавливаем отступы

        self.lineEdit_10 = QtWidgets.QLineEdit(frame_2)
        self.lineEdit_10.setStyleSheet("font-size:24px;\n"
                                       "font: 75 italic 8pt \"Comic Sans MS\";\n"
                                       "color:rgb(67, 74, 84);\n"
                                       "")
        self.lineEdit_10.setObjectName("lineEdit_10")
        frame_layout_2.addWidget(self.lineEdit_10)
        self.lineEdit_10.setAlignment(QtCore.Qt.AlignCenter)

        # Создание надписи и добавление ее в макет
        self.label_10 = QtWidgets.QLabel("Визуализация статистики")
        self.label_10.setStyleSheet("font-size:24px;\n"
                                    "font: 75 \"Comic Sans MS\";\n"
                                    "color:rgb(67, 74, 84);\n"
                                    "")
        frame_layout_2.addWidget(self.label_10)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)

        self.lineEdit_10.setAlignment(QtCore.Qt.AlignCenter)

        # Создание кнопки
        self.pushButton_13_9 = QtWidgets.QPushButton("Запустить вычисления!")
        self.pushButton_13_9.setStyleSheet("background-color: #31A05E;\n"
                                           "font-size:24px;\n"
                                           "font: 75 \"Comic Sans MS\";\n"
                                           "color:rgb(249, 246, 229);\n"
                                           "")

        self.comboboxer = QtWidgets.QComboBox()
        self.comboboxer.setStyleSheet("background-color: #31A05E;\n"
                                           "font-size:24px;\n"
                                           "font: 75 \"Comic Sans MS\";\n"
                                           "color:rgb(249, 246, 229);\n"
                                           "")

        # Установка фиксированного размера для кнопки
        self.pushButton_13_9.setFixedSize(QtCore.QSize(frame_width - 180, 48))
        self.label_10.setFixedSize(QtCore.QSize(frame_width - 40, 48))
        self.lineEdit_10.setFixedSize(QtCore.QSize(frame_width - 180, 48))
        self.comboboxer.setFixedSize(QtCore.QSize(frame_width - 180, 48))

        # Создание растягивающегося элемента для центрирования кнопки по вертикали
        frame_layout_2.addStretch(1)

        # Добавление кнопки в макет
        frame_layout_2.addWidget(self.pushButton_13_9)
        frame_layout_2.addWidget(self.lineEdit_10)
        frame_layout_2.addWidget(self.comboboxer)

        # Установка макета для рамки
        frame_2.setLayout(frame_layout_2)

        button_layout_1_2_1 = QtWidgets.QHBoxLayout()
        button_layout_1_2_1.addStretch(1)  # Добавляем растягивающийся элемент для центрирования
        button_layout_1_2_1.addWidget(self.comboboxer)  # Добавляем кнопку
        button_layout_1_2_1.addStretch(1)  # Добавляем еще один растягивающийся элемент для центрирования
        frame_layout_2.addLayout(button_layout_1_2_1)

        button_layout_1_2 = QtWidgets.QHBoxLayout()
        button_layout_1_2.addStretch(1)  # Добавляем растягивающийся элемент для центрирования
        button_layout_1_2.addWidget(self.lineEdit_10)  # Добавляем кнопку
        button_layout_1_2.addStretch(1)  # Добавляем еще один растягивающийся элемент для центрирования
        frame_layout_2.addLayout(button_layout_1_2)

        button_layout_2_1 = QtWidgets.QHBoxLayout()
        button_layout_2_1.addStretch(1)  # Добавляем растягивающийся элемент для центрирования
        button_layout_2_1.addWidget(self.pushButton_13_9)  # Добавляем кнопку
        button_layout_2_1.addStretch(1)  # Добавляем еще один растягивающийся элемент для центрирования
        frame_layout_2.addLayout(button_layout_2_1)

        for name in self.directories:
            self.comboboxer.addItem(name)

        new_x, new_y = 800, 300 - 130
        self.verticalLayoutWidget.setGeometry(new_x, new_y, self.verticalLayoutWidget.width(),
                                              self.verticalLayoutWidget.height())

        new_x, new_y = 130, 280 - 85
        self.horizontalLayoutWidget.setGeometry(new_x, new_y, self.horizontalLayoutWidget.width(),
                                                self.horizontalLayoutWidget.height())

        new_x, new_y = 300, 290 - 80
        self.verticalLayoutWidget_4.setGeometry(new_x, new_y, self.verticalLayoutWidget_4.width(),
                                                self.verticalLayoutWidget_4.height())

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_1.setText(_translate("MainWindow", "Первый Русский поисковик Парнокопытных!"))
        self.checkBox_2.setText(_translate("MainWindow", "Увеличение точности за счет\nаффинных преобразований"))
        self.checkBox_3.setText(
            _translate("MainWindow", "Ускорение, незначительная потеря\nв точности за счет float16 (вместо float32)"))
        self.checkBox.setText(_translate("MainWindow", "Накладывать кадр под тепловую карту?"))
        self.label_2.setText(_translate("MainWindow", "Сколько пропускать кадров\nпри обработке видео?"))
        self.label_3.setText(_translate("MainWindow", "Пороговая уверенность"))
        self.label_4.setText(_translate("MainWindow", "Коэффициент затухания\nтепловой карты"))
        self.label_5.setText(_translate("MainWindow", "Насколько сильно могут\nперекрываться объекты"))
        self.label_8.setText(_translate("MainWindow", "Масштаб камеры метр\nна пиксель"))
        self.label_6.setText(
            _translate("MainWindow", "Укажите директорию к папке\n(только изображения или только видео)"))
        self.pushButton_12.setText(_translate("MainWindow", "Указать директорию"))
        self.lineEdit.setText(_translate("MainWindow", "Здесь будет указан выбранный путь"))


def _get_dir_name_():
    config_data = load_config()
    script_path = config_data["script_path"]
    path_to_js = os.path.join(script_path, "js")
    if os.path.exists(path_to_js):
        dir_name = os.listdir(path_to_js)
        path_to_dir = [os.path.join(path_to_js, f) for f in os.listdir(path_to_js)]
        dirs = {}
        for index, name in enumerate(dir_name):
            dirs[name] = path_to_dir[index]
        return dirs
    else:
        return ""


class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Назначение переменной логирования
        self.log = setup_logging()

        self.config_data = load_config()
        self.update_config = update_config

        self.pushButton_12.clicked.connect(self.chooseDirectory)
        self.comboboxer.activated.connect(self.on_button_clicker)
        self.pushButton_13.clicked.connect(self._get_linedit_)
        self.pushButton_13_9.clicked.connect(self.on_button_clicked)

    def on_button(self):
        shape_heatmap = self.comboBox.currentText()
        if shape_heatmap == "Круг":
            shape_heatmap = "circle"
        elif shape_heatmap == "Квадрат":
            shape_heatmap = "rectangle"
        self.update_config(self,
                           augment=self.checkBox_2.isChecked(),
                           half=self.checkBox_3.isChecked(),
                           view_img_heatmap=self.checkBox.isChecked(),
                           shape_heatmap=shape_heatmap,
                           vid_stride=self.spinBox.value(),
                           conf=self.doubleSpinBox_2.value(),
                           decay_factor_heatmap=self.doubleSpinBox_3.value(),
                           iou=self.doubleSpinBox.value(),
                           pixel_per_meter=self.doubleSpinBox_4.value())

    def on_button_clicker(self):
        # Получаем текущий индекс выбранного элемента в выпадающем списке
        index = self.comboboxer.currentIndex()

        # Получаем текст выбранного элемента по текущему индексу
        current_text = self.comboboxer.itemText(index)

        # Используем текущий текст для получения пути из self.directories
        if current_text in self.directories:
            directory_path = self.directories[current_text]
            self.lineEdit_10.setText(directory_path)
        else:
            print("Директория не найдена в self.directories")

    def on_button_clicked(self):
        """
        Обработчик нажатия кнопки.
        Возвращает текущий текст из lineEdit_10.
        """
        current_text = self.lineEdit_10.text()
        if current_text:
            path_to_js = current_text
            self.log.info("Началась обработка файлов для построения статистики!")

            if "video" in path_to_js:
                stat = StatObjectDecider(path_to_js=path_to_js)
                deer_count, musk_deer_count, roe_deer_count = stat.get_info_from_video()
            else:
                stat = StatObjectDecider(path_to_js=path_to_js)
                deer_count, musk_deer_count, roe_deer_count = stat.get_info_from_image()

            stat.plot_histogram()

            self.log.info("графики построены!")
        else:
            pass

    def _get_linedit_(self):
        if self.lineEdit:
            self.log.warning("Началась обработка, это может занять некоторое время!")
            source = self.lineEdit.text()
            augment = self.config_data["augment"]
            half = self.config_data["half"]
            view_img_heatmap = self.config_data["view_img_heatmap"]
            shape_heatmap = self.config_data["shape_heatmap"]
            vid_stride = self.config_data["vid_stride"]
            conf = self.config_data["conf"]
            decay_factor_heatmap = self.config_data["decay_factor_heatmap"]
            iou = self.config_data["iou"]
            pixel_per_meter = self.config_data["pixel_per_meter"]
            if pixel_per_meter == 0:
                show_distance = False
            else:
                show_distance = True
            self.log.warning("Загрузка результатов в формате json может занять время!")
            PipeLine(source=source,
                     augment=augment,
                     half=half,
                     view_img_heatmap=view_img_heatmap,
                     shape_heatmap=shape_heatmap,
                     vid_stride=vid_stride,
                     conf=conf,
                     decay_factor_heatmap=decay_factor_heatmap,
                     iou=iou,
                     show_distance=show_distance,
                     pixel_per_meter=pixel_per_meter)
            self.log.info("Обработка завершена!")
            self.directories = _get_dir_name_()
            for name in self.directories:
                if self.comboboxer.findText(name) == -1:
                    self.comboboxer.addItem(name)
        else:
            pass

    def chooseDirectory(self):
        """
        Обработчик нажатия кнопки для выбора директории.
        Открывает диалог выбора директории и устанавливает её путь в lineEdit.
        """
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбрать директорию", QtCore.QDir.currentPath())
        if directory:
            self.lineEdit.setText(directory)
        else:
            pass


# Пример инициализации приложения
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
