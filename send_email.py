import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

remitente = 'manugomezgallego12@gmail.com'
destinatario = 'manugomezgallego12@gmail.com'
password = 'ljxx kjdg awrb iwfs'

msg = MIMEMultipart()
msg['From'] = remitente
msg['To'] = destinatario
msg['Subject'] = 'CDP Jenkins - Compilacion exitosa'

cuerpo = 'Hola Manuela, el job de Jenkins se ejecuto exitosamente. Repositorio: cdp_proyecto. Rama: master. Estado: EXITOSO.'

msg.attach(MIMEText(cuerpo, 'plain'))
servidor = smtplib.SMTP('smtp.gmail.com', 587)
servidor.starttls()
servidor.login(remitente, password)
servidor.sendmail(remitente, destinatario, msg.as_string())
servidor.quit()
print('Correo enviado correctamente')