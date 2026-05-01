## Myanmar Syllable Handwritten Collector
## Vibe Coding with ChatGPT by Ye Kyaw Thu, LU Lab., Myanmar
## Last Updated: 23 Mar 2026
## How to run: python hw_collector.py --help  
## E.g.: python hw_collector.py --file eg-list.txt  

import sys
import os
import time
import json
import argparse
import re

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMessageBox, QInputDialog, QProgressBar
)
from PyQt5.QtGui import QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QPoint, QRect

DATASET_DIR = "dataset"


# -------------------------
# Drawing Canvas
# -------------------------
class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 400)

        self.strokes = []
        self.current_stroke = []
        self.margin = 20

    def get_drawing_rect(self):
        return QRect(
            self.margin,
            self.margin,
            self.width() - 2 * self.margin,
            self.height() - 2 * self.margin
        )

    def inside_area(self, pos):
        return self.get_drawing_rect().contains(pos)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.inside_area(event.pos()):
            self.current_stroke = []
            self.current_stroke.append((event.x(), event.y(), time.time()))
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.inside_area(event.pos()):
            self.current_stroke.append((event.x(), event.y(), time.time()))
            self.update()

    def mouseReleaseEvent(self, event):
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            self.current_stroke = []
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        # boundary
        painter.setPen(QPen(Qt.gray, 2, Qt.DashLine))
        painter.drawRect(self.get_drawing_rect())

        painter.setPen(QPen(Qt.black, 2))

        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                p1 = QPoint(*stroke[i - 1][:2])
                p2 = QPoint(*stroke[i][:2])
                painter.drawLine(p1, p2)

        for i in range(1, len(self.current_stroke)):
            p1 = QPoint(*self.current_stroke[i - 1][:2])
            p2 = QPoint(*self.current_stroke[i][:2])
            painter.drawLine(p1, p2)

    def clear(self):
        self.strokes = []
        self.current_stroke = []
        self.update()

    def save(self, filepath):
        with open(filepath, "w") as f:
            for i, stroke in enumerate(self.strokes):
                f.write(f"STROKE {i+1}\n")
                for (x, y, t) in stroke:
                    f.write(f"{x} {y} {t}\n")
                f.write("\n")


# -------------------------
# Main Window
# -------------------------
class MainWindow(QWidget):
    def __init__(self, text_lines, font_size):
        super().__init__()

        self.setWindowTitle("Myanmar Syllable Handwritten Collector")

        self.text_lines = text_lines
        self.current_index = 0
        self.user_name = None

        self.canvas = DrawingWidget()

        self.font_size = font_size
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.update_font()

        self.user_label = QLabel("Current User: None")

        self.progress = QProgressBar()
        self.progress.setMaximum(len(self.text_lines))

        self.index_input = QLineEdit()
        self.index_input.setPlaceholderText("Jump to line no")

        self.user_list = QListWidget()
        self.load_users()

        # Buttons
        next_btn = QPushButton("Next  [d]")
        prev_btn = QPushButton("Prev  [a]")
        jump_btn = QPushButton("Jump")
        save_btn = QPushButton("Save  [s]")
        clear_btn = QPushButton("Clear  [w]")
        new_user_btn = QPushButton("New User")

        # Connect
        next_btn.clicked.connect(self.next_line)
        prev_btn.clicked.connect(self.prev_line)
        jump_btn.clicked.connect(self.jump_line)
        save_btn.clicked.connect(self.save_sample)
        clear_btn.clicked.connect(self.canvas.clear)
        new_user_btn.clicked.connect(self.create_user)
        self.user_list.itemClicked.connect(self.select_user)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Users"))
        left_layout.addWidget(self.user_list)
        left_layout.addWidget(new_user_btn)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.user_label)
        right_layout.addWidget(self.progress)
        right_layout.addWidget(self.label)
        right_layout.addWidget(self.canvas)

        control_layout = QHBoxLayout()
        control_layout.addWidget(prev_btn)
        control_layout.addWidget(next_btn)
        control_layout.addWidget(self.index_input)
        control_layout.addWidget(jump_btn)
        control_layout.addWidget(save_btn)
        control_layout.addWidget(clear_btn)

        right_layout.addLayout(control_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)

        self.setLayout(main_layout)
        self.setFocusPolicy(Qt.StrongFocus)

        self.update_text()

    # -------------------------
    def update_font(self):
        self.label.setFont(QFont("Noto Sans Myanmar", self.font_size))

    def load_users(self):
        os.makedirs(DATASET_DIR, exist_ok=True)
        self.user_list.clear()
        for u in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR, u)):
                self.user_list.addItem(u)

    # -------------------------
    def update_text(self):
        if self.text_lines:
            text = self.text_lines[self.current_index]
            self.label.setText(f"{self.current_index+1}: {text}")

    def next_line(self):
        if self.current_index < len(self.text_lines) - 1:
            self.current_index += 1
            self.update_text()

    def prev_line(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_text()

    def jump_line(self):
        try:
            idx = int(self.index_input.text()) - 1
            if 0 <= idx < len(self.text_lines):
                self.current_index = idx
                self.update_text()
        except:
            pass

    # -------------------------
    def update_progress(self):
        if not self.user_name:
            return

        user_path = os.path.join(DATASET_DIR, self.user_name)
        written = set()

        for f in os.listdir(user_path):
            m = re.match(r"(\d+)-\d+\.txt", f)
            if m:
                written.add(int(m.group(1)))

        count = len(written)
        total = len(self.text_lines)

        self.progress.setValue(count)
        self.progress.setFormat(f"{count}/{total} ({(count/total)*100:.1f}%)")

    # -------------------------
    def create_user(self):
        name, _ = QInputDialog.getText(self, "User Name", "Enter name:")
        if not name:
            return

        name = name.replace(" ", "_")

        age, _ = QInputDialog.getInt(self, "Age", "Enter age:")
        sex, _ = QInputDialog.getText(self, "Sex", "Enter sex:")
        edu, _ = QInputDialog.getText(self, "Education", "Top education:")

        path = os.path.join(DATASET_DIR, name)
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "user_info.json"), "w") as f:
            json.dump({"name": name, "age": age, "sex": sex, "education": edu}, f)

        self.user_name = name
        self.current_index = 0
        self.user_label.setText(f"Current User: {name}")

        self.load_users()
        self.update_text()
        self.update_progress()

    def select_user(self, item):
        self.user_name = item.text()
        self.user_label.setText(f"Current User: {self.user_name}")

        path = os.path.join(DATASET_DIR, self.user_name)

        max_idx = 0
        for f in os.listdir(path):
            m = re.match(r"(\d+)-\d+\.txt", f)
            if m:
                max_idx = max(max_idx, int(m.group(1)))

        if max_idx > 0:
            self.current_index = max_idx - 1

        self.update_text()
        self.update_progress()

    # -------------------------
    def save_sample(self):
        if not self.user_name:
            QMessageBox.warning(self, "Error", "Select user first")
            return

        path = os.path.join(DATASET_DIR, self.user_name)
        base = str(self.current_index + 1)

        i = 1
        while True:
            fname = f"{base}-{i}.txt"
            fpath = os.path.join(path, fname)
            if not os.path.exists(fpath):
                break
            i += 1

        self.canvas.save(fpath)
        self.canvas.clear()
        self.update_progress()

        QMessageBox.information(self, "Saved", f"{fname}")
        
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_A:
            self.prev_line()
        elif key == Qt.Key_D:
            self.next_line()
        elif key == Qt.Key_S:
            self.save_sample()
        elif key == Qt.Key_W:
            self.canvas.clear()
        elif key == Qt.Key_Space:
            focused = QApplication.focusWidget()
            if isinstance(focused, QPushButton):
                focused.click()


# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Myanmar Syllable Handwritten Collector")
    parser.add_argument("--file", required=True)
    parser.add_argument("--font_size", type=int, default=32)

    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    app = QApplication(sys.argv)
    win = MainWindow(lines, args.font_size)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
