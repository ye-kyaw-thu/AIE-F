## Myanmar Syllable Handwritten Dataset Browser
## Vibe Coding with ChatGPT by Ye Kyaw Thu, LU Lab., Myanmar
## Last Updated: 23 Mar 2026
## How to run: python dataset_browser.py --help
## E.g.: python dataset_browser.py --dataset dataset --textfile eg-list.txt

import sys
import os
import json
import argparse
import re

from PyQt5.QtWidgets import (
    QApplication, QWidget, QListWidget, QLabel,
    QHBoxLayout, QVBoxLayout
)
from PyQt5.QtGui import QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QPoint


# -------------------------
# Stroke Viewer Canvas
# -------------------------
class StrokeViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 400)
        self.strokes = []

    def load_file(self, filepath):
        self.strokes = []

        if not os.path.exists(filepath):
            return

        with open(filepath, "r") as f:
            stroke = []
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("STROKE"):
                    if stroke:
                        self.strokes.append(stroke)
                        stroke = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = int(parts[0]), int(parts[1])
                        stroke.append((x, y))

            if stroke:
                self.strokes.append(stroke)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)

        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                p1 = QPoint(stroke[i - 1][0], stroke[i - 1][1])
                p2 = QPoint(stroke[i][0], stroke[i][1])
                painter.drawLine(p1, p2)


# -------------------------
# Main Window
# -------------------------
class MainWindow(QWidget):
    def __init__(self, dataset_dir, text_lines):
        super().__init__()

        self.setWindowTitle("Handwriting Dataset Browser")

        self.dataset_dir = dataset_dir
        self.text_lines = text_lines

        self.user_list = QListWidget()
        self.file_list = QListWidget()

        self.char_label = QLabel("Character")
        self.char_label.setAlignment(Qt.AlignCenter)
        self.char_label.setFont(QFont("Noto Sans Myanmar", 40))

        self.viewer = StrokeViewer()

        self.load_users()

        # Signals
        self.user_list.currentItemChanged.connect(self.load_files)
        self.file_list.currentItemChanged.connect(self.display_file)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Users"))
        left_layout.addWidget(self.user_list)

        mid_layout = QVBoxLayout()
        mid_layout.addWidget(QLabel("Files"))
        mid_layout.addWidget(self.file_list)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.char_label)
        right_layout.addWidget(self.viewer)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(mid_layout, 1)
        main_layout.addLayout(right_layout, 3)

        self.setLayout(main_layout)

    # -------------------------
    # Load Users
    # -------------------------
    def load_users(self):
        if not os.path.exists(self.dataset_dir):
            return

        self.user_list.clear()
            
        for user in sorted(os.listdir(self.dataset_dir)):
            user_path = os.path.join(self.dataset_dir, user)

            if not os.path.isdir(user_path):
                continue

            self.user_list.addItem(user)

    # -------------------------
    # Load Files per User
    # -------------------------
    def load_files(self):
        self.file_list.clear()

        item = self.user_list.currentItem()
        if not item:
            return

        user = item.text()
        user_path = os.path.join(self.dataset_dir, user)

        files = []
        for fname in os.listdir(user_path):
            if fname.endswith(".txt") and fname != "user_info.json":
                files.append(fname)

        # Sort like 1-1, 1-2, 2-1 ...
        files.sort(key=lambda x: [int(n) for n in re.findall(r'\d+', x)])

        for f in files:
            self.file_list.addItem(f)

    # -------------------------
    # Display Selected File
    # -------------------------
    def display_file(self):
        user_item = self.user_list.currentItem()
        file_item = self.file_list.currentItem()

        if not user_item or not file_item:
            return

        user = user_item.text()
        fname = file_item.text()

        filepath = os.path.join(self.dataset_dir, user, fname)

        # Load strokes
        self.viewer.load_file(filepath)

        # Extract index (e.g., 5-2.txt → 5)
        match = re.match(r"(\d+)-\d+\.txt", fname)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(self.text_lines):
                self.char_label.setText(f"{idx+1}: {self.text_lines[idx]}")
            else:
                self.char_label.setText("Unknown")
        else:
            self.char_label.setText("Unknown")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LipiTK Dataset Browser Tool"
    )
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="Dataset folder (default: dataset)"
    )
    parser.add_argument(
        "--textfile",
        required=True,
        help="Text file (one character per line)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.textfile):
        print("Text file not found")
        sys.exit(1)

    with open(args.textfile, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    app = QApplication(sys.argv)
    window = MainWindow(args.dataset, lines)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
    