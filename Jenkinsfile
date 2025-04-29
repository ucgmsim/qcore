
pipeline {
    agent {
        docker {
            image 'python:3.13'
            // The -u 0 flags means run commands inside the container
            // as the user with uid = 0. This user is, by default, the
            // root user. So it is effectively saying run the commands
            // as root.
            args '-u 0:0'
        }
    }
    stages {
        stage('Installing OS Dependencies') {
            steps {
                echo "[[ Install uv ]]"
                sh """
                    curl -LsSf https://astral.sh/uv/install.sh | sh
                """
            }
        }
        stage('Setting up env') {
            steps {
                echo "[[ Start virtual environment ]]"
                sh """
                    source ~/.local/bin/env sh
                    cd ${env.WORKSPACE}
                    uv venv
                    source .venv/bin/activate
                    uv pip install -e .
                """
            }
        }
        stage('Run regression tests') {
            steps {
                sh """
                    cd ${env.WORKSPACE}
                    source .venv/bin/activate
                    cd qcore/test
                    pytest -s
                """
            }
        }
    }
}
