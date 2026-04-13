pipeline {
    agent any

    triggers {
        // Trigger automático al hacer merge a master
        githubPush()
    }

    stages {

        stage('Clonar Repositorio') {
            steps {
                echo 'Clonando repositorio...'
                git branch: 'master',
                    url: 'https://github.com/manugomez1206/cdp_proyecto.git'
                echo 'Repositorio clonado correctamente'
            }
        }

        stage('Instalar Dependencias') {
            steps {
                echo 'Instalando dependencias...'
                sh 'pip install -r requirements.txt'
                echo 'Dependencias instaladas correctamente'
            }
        }

        stage('Verificar Estructura de Carpetas') {
            steps {
                echo 'Verificando estructura de carpetas...'
                script {
                    def carpetas = [
                        'mlops_pipeline/src',
                        '.github/workflows'
                    ]
                    def archivos = [
                        'mlops_pipeline/src/ft_engineering.py',
                        'mlops_pipeline/src/model_training.py',
                        'mlops_pipeline/src/model_deploy.py',
                        'mlops_pipeline/src/model_monitoring.py',
                        'requirements.txt',
                        'Dockerfile'
                    ]
                    carpetas.each { carpeta ->
                        if (!fileExists(carpeta)) {
                            error "Carpeta no encontrada: ${carpeta}"
                        }
                        echo "Carpeta OK: ${carpeta}"
                    }
                    archivos.each { archivo ->
                        if (!fileExists(archivo)) {
                            error "Archivo no encontrado: ${archivo}"
                        }
                        echo "Archivo OK: ${archivo}"
                    }
                }
                echo 'Estructura de carpetas verificada correctamente'
            }
        }

        stage('Notificación') {
            steps {
                echo 'Enviando notificación...'
                echo 'Pipeline completado exitosamente'
                echo "Repositorio: cdp_proyecto"
                echo "Rama: master"
                echo "Pruebas de estructura: OK"
                echo "Fecha: ${new Date()}"
            }
        }
    }

    post {
        success {
            echo 'Pipeline ejecutado exitosamente'
        }
        failure {
            echo 'Pipeline falló — revisar logs'
        }
    }
}