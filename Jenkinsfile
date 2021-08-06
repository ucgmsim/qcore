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
		docker run --rm  -v ${env.WORKSPACE}:/home/jenkins/qcore --user `id -u`:`id -g` sungeunbae/qcore-ubuntu-minimal bash -c "cd /home/jenkins/qcore;python setup.py install --no-data --user; cd qcore/test; pytest -s;"
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
