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

        stage('Verificar Dependencias') {
            steps {
                echo 'Verificando archivo de dependencias...'
                sh '''
                    if [ -f requirements.txt ]; then
                        echo "requirements.txt encontrado"
                        cat requirements.txt
                    else
                        echo "requirements.txt no encontrado"
                        exit 1
                    fi
                '''
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
                echo 'Estructura verificada correctamente'
            }
        }

        stage('Notificacion') {
            steps {
                echo 'Enviando notificacion por email...'
                sh '''
                    python3 -c "
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

remitente = 'manugomezgallego12@gmail.com'
destinatario = 'manugomezgallego12@gmail.com'
password = 'ljxx kjdg awrb iwfs'

msg = MIMEMultipart()
msg['From'] = remitente
msg['To'] = destinatario
msg['Subject'] = 'CDP Pipeline Jenkins - Compilacion exitosa'

cuerpo = '''
Hola Manuela,

El pipeline de CDP en Jenkins se ejecuto exitosamente.

Detalles:
- Repositorio: cdp_proyecto
- Rama: master
- Pruebas de estructura: OK
- Estado: EXITOSO

Saludos,
Jenkins
'''

msg.attach(MIMEText(cuerpo, 'plain'))
servidor = smtplib.SMTP('smtp.gmail.com', 587)
servidor.starttls()
servidor.login(remitente, password)
servidor.sendmail(remitente, destinatario, msg.as_string())
servidor.quit()
print('Correo enviado correctamente')
"
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline ejecutado exitosamente'
        }
        failure {
            echo 'Pipeline fallo — revisar logs'
        }
    }
}