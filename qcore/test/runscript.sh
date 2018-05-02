echo "from run script"
DIRS='test_*/'
curdir=$PWD
for DIR in $DIRS
do
cd ${curdir}/${DIR}
pwd
echo ${DIR}
pytest --junitxml ${DIR}.xml
done

cp -r ./test_gen_coords.xml /home/aas105/.jenkins/workspace/testtrial/test_gen_coords.xml
cp -r ./test_geo.xml /home/aas105/.jenkins/workspace/testtrial/test_geo.xml
cp -r ./test_srf.xml /home/aas105/.jenkins/workspace/testtrial/test_srf.xml
cp -r ./test_xyts.xml /home/aas105/.jenkins/workspace/testtrial/test_xyts.xml

#py.test --junitxml results.xml qcore/test/test_geo/test_geo.py

#python -m coverage run qcore/test/test_geo/test_geo.py
#python -m coverage xml -o coverage.xml

# py.test --junitxml results.xml
# cp -r ./results.xml /var/lib/jenkins/workspace/pytest_workflow/results.xml