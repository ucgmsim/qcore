DIRS='test_*'
curdir=$PWD
for DIR in $DIRS
do
cd ${curdir}/${DIR}
pytest --junitxml ${DIR}.xml >/dev/null
cp -r ./${DIR}.xml /home/aas105/.jenkins/workspace/qcore-tests/${DIR}.xml
done
LOG=/home/aas105/.jenkins/jobs/qcore-tests/builds/$BUILD_NUMBER/log
cp ${LOG} ${LOG}.html
sed -i 's/$/<br>/' ${LOG}.html
cat ${LOG}.html