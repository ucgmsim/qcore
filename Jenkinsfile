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
		
	//	sh """
	//	source /var/lib/jenkins/py3env/bin/activate
	//	pip install -r requirements.txt
	//	pip install python-coveralls
	//	"""
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
		sh """
		docker run -it -v /var/lib/jenkins/workspace/qcore:/home/root/qcore sungeunbae/qcore-ubuntu-minimal bash -c "cd /home/root/qcore/;python setup.py install; cd qcore/test; pytest -s;"
		"""
            }
        }
        stage('Publish Artifacts') {
            steps {
                echo 'Save the assemblies generated from the compilation' 
            }
        }
    }
}
