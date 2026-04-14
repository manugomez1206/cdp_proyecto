pipeline {
    agent any

    triggers {
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
                sh '''
                    apt-get update -qq
                    apt-get install -y python3 python3-pip -qq
                    pip3 install -r requirements.txt --break-system-packages
                '''
                echo 'Dependencias instaladas correctamente'
            }
        }

        stage('Verificar Estructura de Carpetas') {
            steps {
                echo 'Verificando estructura de carpetas...'
                sh '''
                    python3 -c "
import os
carpetas = ['mlops_pipeline/src', '.github/workflows']
archivos = [
    'mlops_pipeline/src/ft_engineering.py',
    'mlops_pipeline/src/model_training.py',
    'mlops_pipeline/src/model_deploy.py',
    'mlops_pipeline/src/model_monitoring.py',
    'requirements.txt',
    'Dockerfile'
]
for c in carpetas:
    assert os.path.exists(c), f'Carpeta no encontrada: {c}'
    print(f'Carpeta OK: {c}')
for a in archivos:
    assert os.path.exists(a), f'Archivo no encontrado: {a}'
    print(f'Archivo OK: {a}')
print('Estructura verificada correctamente')
"
                '''
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