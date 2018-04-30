echo "from run script"

py.test --junitxml results.xml qcore/test/test_geo.py


python -m coverage run qcore/test/test_geo.py
python -m coverage xml -o coverage.xml
