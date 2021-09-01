pipeline {
    agent any 
    stages {
        stage('Settin up env') {
            steps {
                echo "Start virtual environment"   
                sh """ 
                    pwd
                    env
                    export VENV=/tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv
                    python -m venv $VENV
                    source $VENV/bin/activate
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
                    which python
                    export VENV=/tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/venv
                    source $VENV/bin/activate
                    which python
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
//                rm -rf /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}
            """
        }
    }
}
