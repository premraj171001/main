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
                bat 'dir'
            }
        }
        stage('Building docker image'){
            steps{
                               
                //deleting old image if exist
                bat 'docker rmi healthbridgeimg:v1.1 >nul 2>&1 || docker build -t healthbridgeimg:v1.1 .'

            }
        }
        stage('building a volume'){
            steps{
                bat 'docker volume inspect HBDB >nul 2>&1 || docker volume create HBDB'
            }
        }
        stage('building a network'){
            steps{
                bat 'docker network inspect HBDB >nul 2>&1 || docker network create HBDB'
            }
        }
        
        stage('deploy the docker image'){
            steps{
 
                //deleting old container
                bat 'docker rm -f healthbridgecont || true'

                //running the container
                bat 'docker run -d -p 3012:3012 -v HBDB --network HBDB --name healthbridgecont healthbridgeimg:v1.1'

            }
        }
    }
}
