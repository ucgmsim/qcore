DIRS='test_*'
curdir=$PWD
for DIR in $DIRS
do
cd ${curdir}/${DIR}
pytest --junitxml ${DIR}.xml
cp -r ./${DIR}.xml /home/tester/.jenkins/workspace/qcore-tests/${DIR}.xml
done
LOG=/home/tester/.jenkins/jobs/qcore-tests/builds/$BUILD_NUMBER/log
sed -i 's/$/<br>/' ${LOG}

