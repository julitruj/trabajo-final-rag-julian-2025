import json
import boto3
import os
import time
import urllib.parse
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# --- Configuración Inicial ---
s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')

# Configuración específica para OpenSearch Serverless
endpoint = os.environ['OPENSEARCH_ENDPOINT']
region = os.environ.get('AWS_REGION', 'us-east-1')

# Extraer el host del endpoint
if endpoint.startswith('https://'):
    opensearch_host = endpoint.replace('https://', '')
else:
    opensearch_host = endpoint

print(f"Configurando OpenSearch Serverless - Host: {opensearch_host}, Region: {region}")

# Configurar autenticación para Serverless con configuración específica
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, 'aoss')

# Cliente OpenSearch configurado específicamente para Serverless
client = OpenSearch(
    hosts=[{'host': opensearch_host, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30,
    max_retries=0,
    retry_on_timeout=False
)
index_name = 'rag-final-collection'

def lambda_handler(event, context):
    key = "unknown"  # Inicializar para evitar error en el except
    
    try:
        print(f"🚀 Iniciando procesamiento...")
        print(f"Evento recibido: {json.dumps(event, indent=2)}")
        
        # SALTEAR COMPLETAMENTE el test de conexión
        print("📡 Saltando test de conexión - directamente al procesamiento")
        
        # Saltear creación de índice - asumimos que existe
        print("📝 Asumiendo que el índice rag-final-collection ya existe")
        
        # Pausa de 2 segundos para evitar la condición de carrera de S3
        print("⏳ Esperando 2 segundos para evitar condición de carrera...")
        time.sleep(2)
        
        # 1. Obtener el bucket y el archivo .txt del evento S3
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        
        print(f"📁 Procesando archivo: s3://{bucket}/{key}")
        
        if not key.startswith('trusted/'):
            print(f"⚠️ Archivo {key} no está en trusted/, saltando...")
            return {'statusCode': 200, 'body': json.dumps('Archivo no está en trusted/')}
        
        # 2. Leer el contenido del archivo .txt desde S3
        print(f"📖 Leyendo contenido del archivo...")
        s3_object = s3.get_object(Bucket=bucket, Key=key)
        text_content = s3_object['Body'].read().decode('utf-8')
        
        # Verificar que el contenido no esté vacío
        if not text_content.strip():
            print("⚠️ El archivo está vacío, saltando...")
            return {'statusCode': 200, 'body': json.dumps('Archivo vacío')}
        
        print(f"📄 Contenido leído: {len(text_content)} caracteres")
        
        # 3. Generar el embedding con Amazon Bedrock Titan
        print("🧠 Generando embedding con Bedrock Titan...")
        # Limitar el texto para Bedrock (máximo ~25000 tokens)
        text_for_embedding = text_content[:8000] if len(text_content) > 8000 else text_content
        
        body = json.dumps({"inputText": text_for_embedding})
        response = bedrock.invoke_model(
            body=body, modelId='amazon.titan-embed-text-v1',
            accept='application/json', contentType='application/json'
        )
        response_body = json.loads(response['body'].read())
        embedding = response_body['embedding']
        print(f"✅ Embedding generado. Dimensiones: {len(embedding)}")
        
        # 4. Indexar el documento y su vector en OpenSearch
        print(f"💾 Indexando documento en OpenSearch...")
        document = {
            'text': text_content,
            'vector_field': embedding,
            'source_file': key,
            'timestamp': int(time.time() * 1000)  # timestamp en milisegundos
        }
        
        print(f"📄 Preparando documento para indexación...")
        
        # Indexar directamente sin verificar si el índice existe
        # OpenSearch Serverless no soporta IDs personalizados ni refresh=True
        response = client.index(
            index=index_name,
            body=document
        )
        print(f"✅ Documento indexado exitosamente!")
        print(f"📊 Respuesta de OpenSearch: {response}")
        
        return {
            'statusCode': 200, 
            'body': json.dumps({
                'message': 'Procesamiento completado exitosamente',
                'file': key,
                'opensearch_response': response
            })
        }
        
    except Exception as e:
        error_msg = f"❌ Error procesando archivo {key}: {str(e)}"
        print(error_msg)
        
        import traceback
        print(f"🔍 Traceback completo:")
        print(traceback.format_exc())
        
        # Re-lanzar la excepción para que Lambda la marque como fallo
        raise e
