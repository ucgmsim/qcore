pipeline {
    agent any 
    stages {
	stage('Check Docker image') {
	    steps {
		echo "Checking if the docker image is available (logic not implemented)"
		sh """
		docker images sungeunbae/qcore-ubuntu-minimal -q
		"""
	    }
	}
        stage('Run regression tests') {
            steps {
                echo 'Run pytest through docker' 
		sh """
		docker run  -v ${env.WORKSPACE}:/home/root/qcore sungeunbae/qcore-ubuntu-minimal bash -c "cd /home/root/test; cp -rf /home/root/qcore/* .;python setup.py install; cd qcore/test; pytest -s;"
		"""
            }
        }
    }

    post {
	always {
                echo 'Tear down the environments'
		sh """
		docker container prune -f
		"""
            }
    }
}
