python -m venv venv
.\venv\Scripts\activate
pip list
pip install --upgrade pip setuptools wheel
pip install --no-deps -r requirements-coeiroink-no-deps.txt
pip install -r requirements-coeiroink.txt
pip install -r requirements-dev.in
python generate_licenses.py > licenses.json
pip install pyinstaller
pyinstaller run.py
mkdir dist/run/espnet
cp venv/Lib/site-packages/espnet/version.txt dist/run/espnet/
mkdir dist/run/librosa/util/example_data
cp venv/Lib/site-packages/librosa/util/example_data/registry.txt dist/run/librosa/util/example_data/
cp venv/Lib/site-packages/librosa/util/example_data/index.json dist/run/librosa/util/example_data/
cp default.csv dist/run/
