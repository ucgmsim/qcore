pwd
cd /qcore/qcore/test
echo "from run script"
DIRS='test_*'
curdir=$PWD
for DIR in $DIRS
do
cd ${curdir}/${DIR}
pwd
echo ${DIR}
pytest --junitxml ${DIR}.xml
#cp -r ./${DIR}.xml /home/aas105/.jenkins/workspace/fromlocal/${DIR}.xml
done

#py.test --junitxml results.xml qcore/test/test_geo/test_geo.py

#python -m coverage run qcore/test/test_geo/test_geo.py
#python -m coverage xml -o coverage.xml

# py.test --junitxml results.xml
# cp -r ./results.xml /var/lib/jenkins/workspace/pytest_workflow/results.xml