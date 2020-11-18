pipeline {
    agent any 
    stages {
        stage('Static Analysis') {
            steps {
                echo 'Run the static analysis to the code' 
            }
        }
        stage('Compile') {
            steps {
                echo 'Compile the source code' 
		sh 'pip install -r requirements.txt'
		sh 'pip install python-coverall'
            }
        }
        stage('Security Check') {
            steps {
                echo 'Run the security check against the application' 
            }
        }
        stage('Run Unit Tests') {
            steps {
                echo 'Run unit tests from the source code' 
            }
        }
        stage('Run Integration Tests') {
            steps {
                echo 'Run only crucial integration tests from the source code' 
		sh 'cd qcore'
		sh 'python setup.py install'
		sh 'cd qcore/test'
		sh 'pytest -s'
            }
        }
        stage('Publish Artifacts') {
            steps {
                echo 'Save the assemblies generated from the compilation' 
            }
        }
    }
}
