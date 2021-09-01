pipeline {
    agent any 
    stages {
        stage('Settin up env') {
            steps {
                echo "Start virtual environment"   
                sh """ 
                    pwd
                    env
                    source /var/lib/jenkins/py3env/bin/activate
                    mkdir -p /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}
# I don't know how to create a variable within Jenkinsfile
#                    export virtenv=/tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv

                    which python
                    python -m venv /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv
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
