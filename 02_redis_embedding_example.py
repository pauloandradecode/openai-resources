import redis
import openai
import getpass
import time
import numpy as np
import json


# Varianles globales
openai.api_key = getpass.getpass()
model = 'text-embedding-ada-002'

# creamos un cliente de redis
class RedisClient:
    def __init__(self, host='localhost', port=6379):
        self.client = redis.Redis(host=host, port=port)

    def set(self, key, value):
        try:
            return self.client.set(key, json.dumps(value))
        except ConnectionError:
            print("Connection Error")

    def get(self, key):
        try:
            return json.loads(self.client.get(key))
        except ConnectionError:
            print("Connection Error")

    def delete(self, key):
        try:
            print("Deleting key: " + key)
            return self.client.delete(key)
        except ConnectionError:
            print("Connection Error")

    def keys(self, pattern='*'):
        try:
            return self.client.keys(pattern)
        except ConnectionError:
            print("Connection Error")

    
    def exists(self, key):
        try:
            return self.client.exists(key)
        except ConnectionError:
            print("Connection Error")
            

    def flushall(self):
        try:
            return self.client.flushall()
        except ConnectionError:
            print("Connection Error")

    def flushdb(self):
        try:
            return self.client.flushdb()
        except ConnectionError:
            print("Connection Error")


# retrieve the embedding from the redis cache
# if it doesn't exist, create it and store it in the cache
# then return it

def get_embedding(text):
    # create a redis client
    redis_client = RedisClient()

    # redis_client.delete(text)

    # if the embedding doesn't exist in the cache
    if not redis_client.exists(text):
        # create the embedding
        time.sleep(1)
        print('Creating embedding for: ' + text)
        
        response = openai.Embedding.create(
            input = text,
            model = model
        )

        # print(response['data'][0]['embedding'])
        embeddings = response['data'][0]['embedding']
        
        # store the embedding in the cache
        redis_client.set(text, embeddings)
    # return the embedding
    else:
        embeddings = redis_client.get(text)
    return embeddings


# Ejemplo de funcionamiento
embeddings = get_embedding("This is a test")
print(embeddings)

# openai.Engine.list() 