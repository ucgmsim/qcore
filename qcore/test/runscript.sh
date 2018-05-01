echo "from run script"
DIRS='test_*/'
curdir=$PWD
for DIR in $DIRS
do
cd ${curdir}/${DIR}
pwd
echo ${DIR}
pytest -s -v
done


#py.test --junitxml results.xml qcore/test/test_geo/test_geo.py

#python -m coverage run qcore/test/test_geo/test_geo.py
#python -m coverage xml -o coverage.xml

# py.test --junitxml results.xml
# cp -r ./results.xml /var/lib/jenkins/workspace/pytest_workflow/results.xml