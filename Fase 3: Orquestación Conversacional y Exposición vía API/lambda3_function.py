import json
import boto3
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Configuración Inicial ---
# Variables de entorno y clientes de AWS
opensearch_host = os.environ['OPENSEARCH_ENDPOINT'].replace('https://', '')
opensearch_index = os.environ['OPENSEARCH_INDEX']
region = os.environ.get('AWS_REGION', 'us-east-1') # Obtener la región del entorno

# Crear cliente de Bedrock
bedrock_client = boto3.client('bedrock-runtime', region_name=region)

def get_langchain_components():
    """Inicializa y devuelve los componentes de LangChain necesarios."""
    
    # Modelo de embeddings para convertir la pregunta en un vector
    embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")
    
    # Autenticación para OpenSearch
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, 'aoss')

    # Vector store para buscar en nuestra base de datos OpenSearch
    vector_store = OpenSearchVectorSearch(
        opensearch_url=f"https://{opensearch_host}:443",
        index_name=opensearch_index,
        embedding_function=embeddings,
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    
    # Modelo de lenguaje para generar la respuesta final
    llm = Bedrock(
        client=bedrock_client, 
        model_id="anthropic.claude-v2:1", # Usamos una versión específica de Claude por estabilidad
        model_kwargs={"max_tokens_to_sample": 1000, "temperature": 0.1}
    )
    
    # Memoria para que el chatbot recuerde el historial
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    
    # Cadena RAG conversacional que une todo
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={'k': 3}), # Recuperar los 3 documentos más relevantes
        memory=memory,
        return_source_documents=True # Devolver los documentos fuente para validación
    )
    
    return chain

def lambda_handler(event, context):
    print(f"Evento recibido: {event}")
    
    try:
        # Extraer la pregunta del cuerpo de la petición de API Gateway
        body = json.loads(event.get('body', '{}'))
        question = body.get('question')
        if not question:
            return {'statusCode': 400, 'body': json.dumps('No se encontró ninguna pregunta.')}

        print(f"Pregunta del usuario: {question}")
        
        qa_chain = get_langchain_components()
        
        print("Ejecutando la cadena RAG...")
        result = qa_chain.invoke({"question": question})
        answer = result.get('answer', 'No se pudo generar una respuesta.')
        
        # Formatear los documentos fuente para incluirlos en la respuesta
        sources = []
        if result.get('source_documents'):
            for doc in result['source_documents']:
                sources.append({
                    "source": doc.metadata.get('source_file', 'Fuente desconocida'),
                    "content_preview": doc.page_content[:250] + "..."
                })
        
        print(f"Respuesta generada: {answer}")
        
        # Devolver la respuesta en el formato que espera API Gateway
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' # Permite que cualquier web llame a esta API
            },
            'body': json.dumps({'answer': answer, 'sources': sources})
        }

    except Exception as e:
        print(f"Error al ejecutar la cadena RAG: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error interno del servidor: {str(e)}')
        }
