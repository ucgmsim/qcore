DIRS='test_*'
curdir=$PWD
for DIR in $DIRS
do
printf '<br><br><br>'
cd ${curdir}/${DIR}
pytest --junitxml ${DIR}.xml
cp -r ./${DIR}.xml /home/aas105/.jenkins/workspace/qcore-tests/${DIR}.xml
done
printf '<br><br><br>'

#py.test --junitxml results.xml qcore/test/test_geo/test_geo.py

#python -m coverage run qcore/test/test_geo/test_geo.py
#python -m coverage xml -o coverage.xml

# py.test --junitxml results.xml
# cp -r ./results.xml /var/lib/jenkins/workspace/pytest_workflow/results.xml