
pipeline {
    agent {
        docker { image 'python:3.13' }
    }
    stages {
        stage('Installing OS Dependencies') {
            steps {
                echo "[[ Install GMT ]]"
                sh """
                   apt-get update
                   apt-get install -y gmt libgmt-dev libgmt6 ghostscript
                """
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
