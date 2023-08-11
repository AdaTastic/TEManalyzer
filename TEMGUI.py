import sys, os
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QLabel, QSlider, QWidget, QPushButton, QColorDialog, QFileDialog, QGridLayout
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QMovie, qRgb
from PyQt5.QtCore import Qt, QPointF
import preview, output
import dm3reader
import numpy as np
import cv2
import h5py

class TEMAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Editor")
        self.setGeometry(100, 100, 600, 400)

        self.layout = QGridLayout()

        # image editor/viewer
        self.image_label = QLabel()
        self.image_filenames = []
        self.image_list = []
        self.current_image_index = 0
        self.image = None
        self.paint_mask_list = []
        self.layout.addWidget(self.image_label,2,0,5,5)

        # preview viewer
        self.preview_window = QLabel()
        self.preview_list = []
        self.current_preview_index = 0
        self.preview = None
        self.layout.addWidget(self.preview_window,2,5,5,5)

        # blank image template
        self.blank1 = QLabel()
        self.blank2 = QLabel()
        self.load_blank()
        self.layout.addWidget(self.blank1,2,0,5,5)
        self.layout.addWidget(self.blank2,2,5,5,5)

        self.canvas_widget = QWidget()
        self.canvas_widget.setLayout(self.layout)
        self.setCentralWidget(self.canvas_widget)

        # load images
        open_action = QAction("Open Folder", self)
        open_action.triggered.connect(self.open_folder)
        self.toolbar = self.addToolBar("Open")
        self.toolbar.addAction(open_action)

        # prev button
        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.layout.addWidget(self.prev_button,0,0,2,1)

        # next button
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.show_next_image)
        self.layout.addWidget(self.next_button,0,1,2,1)

        # brush size
        self.brush_size_up = QPushButton("△", self)
        self.brush_size_up.clicked.connect(self.increase_brush_size)
        self.layout.addWidget(self.brush_size_up,0,2)

        self.brush_size_down = QPushButton("▽", self)
        self.brush_size_down.clicked.connect(self.decrease_brush_size)
        self.layout.addWidget(self.brush_size_down,1,2)

        # contrast slider
        self.slider1 = QSlider(Qt.Horizontal, self)
        self.slider1.setRange(-100, 300)
        self.slider1.setValue(100)
        self.slider1.valueChanged.connect(self.update_contrast)
        self.layout.addWidget(self.slider1,0,3)

        # brightness slider
        self.slider2 = QSlider(Qt.Horizontal, self)
        self.slider2.setRange(-100, 100)
        self.slider2.setValue(0)
        self.slider2.valueChanged.connect(self.update_contrast)
        self.layout.addWidget(self.slider2,0,4)

        # reset edits button
        self.reset = QPushButton("reset", self)
        self.reset.clicked.connect(self.reset_image)
        self.layout.addWidget(self.reset,1,3)

        # save edits button
        self.save = QPushButton("save/replace", self)
        self.save.clicked.connect(self.save_image)
        self.layout.addWidget(self.save,1,4,1,1)

        # gen output button
        self.output = QPushButton("Analyze All Images (full)", self)
        self.output.clicked.connect(self.gen_output)
        self.layout.addWidget(self.output,0,7,2,1)

        # generate preview
        self.preview_button = QPushButton("Show/Gen Preview", self)
        self.preview_button.clicked.connect(self.show_preview)
        self.layout.addWidget(self.preview_button,0,5,2,1)

        # generate histogram
        self.preview_histogram = QPushButton("preview histogram", self)
        self.preview_histogram.clicked.connect(self.show_preview_histogram)
        self.layout.addWidget(self.preview_histogram,0,6,2,1)
        
        # generation loading icon
        # loading_gif_path = "icons\loading.gif"  # Replace with the path to your GIF image
        # self.movie = QMovie(loading_gif_path)
        # self.loading_label = QLabel()
        # self.loading_label.setMovie(self.movie)
        # self.layout.addWidget(self.loading_label,2,5,2,2)
        # self.loading_label.raise_()

        # brush variables
        self.brush_color = Qt.white
        self.brush_size = 10

# loading and showing images code ---------------------------------------------------------------------------------------------
    def load_blank(self):
        blank = np.zeros((500, 500), dtype=np.uint8)
        height, width = blank.shape[:2]
        blank = QImage(blank, width, height, width, QImage.Format_Grayscale8)
        self.blank1.setPixmap(QPixmap.fromImage(blank))
        self.blank2.setPixmap(QPixmap.fromImage(blank))

    def open_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder", "", QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if folder_path:
            self.image_filenames = []
            self.image_list = []
            self.current_image_index = 0
            self.image = None
            self.load_images_from_folder(folder_path)

    def load_images_from_folder(self, folder_path):
        self.image_filenames = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith((".png", ".jpg", "dm3"))]
        for file in self.image_filenames:
            if file.lower().endswith(".dm3"):
                dm3_array = dm3reader.getDM3Image(file)
                height, width = dm3_array.shape[:2]
                self.image_list.append(dm3_array)
                self.paint_mask_list.append(QImage(width, height, QImage.Format_RGBA8888))
            else:
                image_array = np.array(Image.open(file).convert("RGBA"))
                height, width = image_array.shape[:2]
                self.image_list.append(image_array)
                self.paint_mask_list.append(QImage(width, height, QImage.Format_RGBA8888))

        self.preview_list = [None] * len(self.image_list)
        if self.image_list:
            self.current_image_index = 0
            self.image = self.image_list[self.current_image_index]
            self.show_current_image()

    def show_current_image(self):
        if self.image_list:
            self.blank1.setVisible(False) 
            painted_img = self.overlay_paint(self.paint_mask_list[self.current_image_index],self.image )
            self.image_label.setPixmap(painted_img.scaledToWidth(500, Qt.SmoothTransformation))

    def overlay_paint(self, image: QImage, array: np.array):
        array_height, array_width = array.shape[:2]
        overlay_image = QImage(array.data, array_width, array_height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(overlay_image)
        painter = QPainter(pixmap)
        painter.drawImage(0, 0, image)
        painter.end()
        # result_image = pixmap.toImage()
        return pixmap

    def show_previous_image(self):
        if self.image_list:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
            self.image = self.image_list[self.current_image_index]
            self.slider1.setValue(100)
            self.slider2.setValue(0)
            self.show_current_image()
            self.show_current_preview()

    def show_next_image(self):
        if self.image_list:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
            self.image = self.image_list[self.current_image_index]
            self.slider1.setValue(100)
            self.slider2.setValue(0)
            self.show_current_image()
            self.show_current_preview()

    def qimage_to_array(self, qimage: QImage) -> np.ndarray:
        width = qimage.width()
        height = qimage.height()
        format = qimage.format()

        if format == QImage.Format_ARGB32 or format == QImage.Format_RGBA8888:
            dtype = np.uint8
            channels = 4
        elif format == QImage.Format_RGB32:
            dtype = np.uint8
            channels = 3
        else:
            raise ValueError("Unsupported QImage format")

        buffer = qimage.bits()
        buffer.setsize(qimage.byteCount())
        array = np.frombuffer(buffer, dtype=dtype).reshape((height, width, channels))
        return array

# editing, contrast, brightness code  ------------------------------------------------------------------------------------------------------------
    def update_contrast(self):
        if not self.image_list:
            return
        self.image = self.image_list[self.current_image_index]
        contrast = self.slider1.value()/100
        brightness = self.slider2.value()
        bgr_array = self.image[..., :3]
        contrast_adjusted_bgr = cv2.convertScaleAbs(bgr_array, alpha=contrast, beta=brightness)
        self.image = np.dstack((contrast_adjusted_bgr, self.image[..., 3]))
        self.show_current_image()
        print(contrast)
        print(brightness)

    def reset_image(self):
        if not self.paint_mask_list:
            return
        self.paint_mask_list[self.current_image_index] = QImage(self.paint_mask_list[self.current_image_index].size(), QImage.Format_RGBA8888)
        self.image = self.image_list[self.current_image_index]
        self.slider1.setValue(100)
        self.slider2.setValue(0)
        self.update_contrast()
        self.show_current_image()

    def save_image(self):
        if not self.image_filenames:
            return
        
        paint_array = self.qimage_to_array(self.paint_mask_list[self.current_image_index])
        image = Image.fromarray(self.image)
        image = image.convert('RGBA')
        paint = Image.fromarray(paint_array)
        data = Image.alpha_composite(image, paint)

        # if dm3, replace with .hdf5 containing info
        if self.image_filenames[self.current_image_index].lower().endswith(".dm3"):
            pixelUnit, PixelSize = dm3reader.getDM3scale(self.image_filenames[self.current_image_index])
            edited_image = {}
            edited_image['data'] = np.array(data)
            edited_image['pixelUnit'] = pixelUnit
            edited_image['pixelSize'] = PixelSize       
            #saving
            file_path = os.path.splitext(self.image_filenames[self.current_image_index])[0] + ".hdf5"
            self.image_filenames[self.current_image_index] = file_path
            
            with h5py.File(file_path, "w") as f:
                group = f.create_group("my_group")
                for key, value in edited_image.items():
                    group.create_dataset(key, data=value)
            print(type(pixelUnit))
            
        else:
            file_path = os.path.splitext(self.image_filenames[self.current_image_index])[0] + ".jpg"
            self.image_filenames[self.current_image_index] = file_path
            data.save(file_path)

# Preview code ------------------------------------------------------------------------------------------------------------
    def show_preview(self):
        if not self.image_list:
            return
        image_path = self.image_filenames[self.current_image_index]
        preview_folder = os.path.join(os.path.dirname(image_path), "preview")
        preview_image_filename = os.path.splitext(os.path.basename(image_path))[0] + "_resized_preview.jpg"
        preview_image_filepath = os.path.join(preview_folder, preview_image_filename)
        # Check if the target directory exists
        if not os.path.exists(preview_folder):
            # Create the directory (including any missing parent directories)
            os.makedirs(preview_folder)

        if os.path.exists(preview_image_filepath):
            self.preview_list[self.current_image_index] = QImage(preview_image_filepath)
            self.show_current_preview()
            return
        
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        model_path = os.path.join(current_directory,"model/model_final.pth")
        config_path = os.path.join(current_directory,"model/config.yaml")
        results, resized_img_path = preview.runPreviewPredict(image_path,model_path,config_path)
        results_path = preview.filterParticles(resized_img_path,results,image_path)
        preview.genPreview(resized_img_path,results_path)

        self.preview_list[self.current_image_index] = QImage(preview_image_filepath)
        self.show_current_preview()

    def show_current_preview(self):
        if self.preview_list[self.current_image_index] != None:
            self.blank2.setVisible(False)
            self.preview = self.preview_list[self.current_image_index]
            scaled_image = self.preview.scaledToWidth(500, Qt.SmoothTransformation)
            self.preview_window.setPixmap(QPixmap.fromImage(scaled_image))

    def show_preview_histogram(self):
        if not self.image_filenames:
            return
        image_path = self.image_filenames[self.current_image_index]
        preview_folder = os.path.join(os.path.dirname(image_path), "preview")
        results_name = os.path.splitext(os.path.basename(image_path))[0] + "_resized_results.txt"
        preview_results_path = os.path.join(preview_folder,results_name)
        if not os.path.exists(preview_results_path):
            return
        preview.genHistogram(preview_results_path)
        
# Output code ------------------------------------------------------------------------------------------------------------
    def gen_output(self):
        if not self.image_list:
                return
        for image_filename in self.image_filenames:
            image_path = image_filename
            output_folder = os.path.join(os.path.dirname(image_path), "output")
            results_path = os.path.splitext(os.path.basename(image_path))[0] + "_resized_results.txt"
            output_results_path = os.path.join(output_folder, results_path)
            # Check if the target directory exists
            if not os.path.exists(output_folder):
                # Create the directory (including any missing parent directories)
                os.makedirs(output_folder)
            
            current_file_path = os.path.abspath(__file__)
            current_directory = os.path.dirname(current_file_path)
            model_path = os.path.join(current_directory,"model/model_final.pth")
            config_path = os.path.join(current_directory,"model/config.yaml")
            results, resized_img_path = output.runOutputPredict(image_path,model_path,config_path)
            results_path = output.filterParticles(resized_img_path,results,image_path)
            output.genOutputImage(resized_img_path,results_path)
            output.writeData(results_path)

# Brush and Painting code ------------------------------------------------------------------------------------------------------------
    def increase_brush_size(self):
        self.brush_size += 10

    def decrease_brush_size(self):
        self.brush_size -= 10

    def edit_image(self):
        if self.image_list:
            self.painter = QPainter(self.paint_mask_list[self.current_image_index])
            self.painter.setPen(QColor(self.brush_color))
            self.painter.setBrush(QColor(self.brush_color))
            self.painter.drawEllipse(self.brush_pos, self.brush_size, self.brush_size)
            self.painter.end()
            # self.image_label.setPixmap(QPixmap.fromImage(self.image))
            self.show_current_image()

    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.image_list:
            self.brush_pos = self.adjusted_pos(event.pos())
            print(self.brush_pos)
            self.edit_image()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.image_list:
            self.brush_pos = self.adjusted_pos(event.pos())
            # self.brush_pos = self.brush_pos/2048 * 500
            self.edit_image()

    def adjusted_pos(self, original_pos):
        # Modify the original_pos here as needed and return the adjusted position
        # For example, to move the brush position down by 20 pixels, you can do the following:
        height,width = self.image_list[self.current_image_index].shape[:2]
        adjusted_x = original_pos.x()/500*width
        adjusted_y = original_pos.y()/500*height
        return QPointF(adjusted_x, adjusted_y)
    
    def start_load(self):
        self.movie.start()

    def stop_load(self):
        self.movie.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TEMAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
