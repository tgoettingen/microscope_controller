python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

pyinstaller --noconfirm --clean --onefile --windowed --name MicroscopeController --paths . --hidden-import PyQt6.sip --collect-submodules core --collect-submodules devices --collect-submodules gui --collect-submodules utils --exclude-module PyQt5 --exclude-module PySide2 --exclude-module PySide6 --exclude-module PyQt4 --exclude-module qtpy gui/mainwindow.py