pipeline{
    agent any
    stages{
        stage('git data'){
            steps{
                git branch: 'main', url: 'https://github.com/premraj171001/main.git'
            }
        }
        stage('list the items'){
            steps{
                sh 'ls'
            }
        }
        stage('Building docker image'){
            steps{
                sh 'docker build -t healthbridgeimg:v1.1 .'
            }
        }
        stage('deploy the docker image'){
            steps{
                sh 'docker run -d -p 3012:3012 --name healthbridgecont healthbridgeimg:v1.1'

            }
        }
    }
}