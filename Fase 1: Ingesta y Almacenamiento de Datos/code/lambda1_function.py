import json
import boto3
import fitz  # Esta es la librería PyMuPDF que importamos desde la capa
import urllib.parse

# Creamos un cliente de S3 para interactuar con el servicio
s3 = boto3.client('s3')

import urllib.parse

def lambda_handler(event, context):
    # 1. Obtenemos el nombre del bucket y el archivo (key) del evento que nos envía S3
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

    # Una salvaguarda para evitar bucles infinitos
    if not key.startswith('raw/'):
        print(f"El archivo {key} no está en la carpeta 'raw/'. Saliendo.")
        return

    try:
        # 2. Descargamos el PDF desde la carpeta 'raw/' a un directorio temporal de Lambda (/tmp/)
        # El directorio /tmp/ es el único lugar donde Lambda nos permite escribir archivos
        download_path = f'/tmp/{key.split("/")[-1]}'
        print(f"Descargando s3://{bucket}/{key} a {download_path}")
        s3.download_file(bucket, key, download_path)

        # 3. Usamos PyMuPDF (fitz) para abrir el PDF y extraer todo su texto
        print("Extrayendo texto del PDF...")
        doc = fitz.open(download_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        print("Extracción de texto completada.")

        # 4. Definimos la ruta de destino en la carpeta 'trusted/'
        # Cambiamos 'raw/' por 'trusted/' y la extensión '.pdf' por '.txt'
        # Obtenemos solo el nombre del archivo original, sin la ruta raw/ y sin la extensión .pdf
        original_filename = key.split('/')[-1].rsplit('.', 1)[0]

        # Reemplazamos los espacios por guiones bajos para crear un nombre seguro
        sanitized_filename = original_filename.replace(' ', '_')

        # Creamos la nueva ruta de destino en la carpeta trusted/ con el nombre sanitizado
        destination_key = f"trusted/{sanitized_filename}.txt"

        # 5. Subimos el texto extraído como un nuevo archivo .txt a la carpeta 'trusted/'
        print(f"Subiendo texto a s3://{bucket}/{destination_key}")
        s3.put_object(Bucket=bucket, Key=destination_key, Body=full_text, ContentType='text/plain')

        return {
            'statusCode': 200,
            'body': json.dumps(f'Procesado {key} exitosamente y subido a {destination_key}')
        }

    except Exception as e:
        print(f"Error procesando el archivo {key}: {e}")
        raise e
