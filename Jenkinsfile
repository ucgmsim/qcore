pipeline {
    agent any 
    stages {
        stage('Checking env') {
            steps {
                echo "Check the environment"
                sh """
                    pwd
                    env
                """
            }
        }
        stage('Settin up env') {
            steps {
                echo "Start virtual environment"   
                sh """
# Each stage needs custom setting done again. By default /biin/python is used.
                    source /var/lib/jenkins/py3env/bin/activate
                    mkdir -p /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}
# I don't know how to create a variable within Jenkinsfile (please let me know)
#                   export virtenv=/tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv
                    python -m venv /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv
# activate new virtual env
                    source /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv/bin/activate
                    cd ${env.WORKSPACE}
                    echo "Install dependencies"
                    pip install -r requirements.txt
                    
                """
            }
        }
        stage('Run regression tests') {
            steps {
                echo 'Run pytest' 
                sh """
# activate virtual environment again
                    source /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv/bin/activate
                    cd ${env.WORKSPACE}
                    python setup.py install --no-data
                    cd qcore/test
                    pytest -s
                """
            }
        }
    }

    post {
        always {
            echo 'Tear down the environments'
            sh """
                rm -rf /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}
            """
        }
    }
}
