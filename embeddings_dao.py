import pymongo


class EmbeddingsDao:
    def __init__(self):
        self.username = 'root'
        self.password = 'root'
        self.db_name = 'AttendanceSystem'
        self.connection_url = f"mongodb+srv://{self.username}:{self.password}@cluster0.oxlmy.mongodb.net/{self.db_name}?retryWrites=true&w=majority"
        self.client = pymongo.MongoClient(self.connection_url)
        self.db = self.client[self.db_name]
        self.collection_name = 'Employees'
        self.collection = self.db[self.collection_name]

    def get_embedding(self, name):
        query = dict()
        query['name'] = name
        res = self.collection.find_one(query)
        return res['embeddings']

    def save_embedding(self, name, embedding):
        query = dict()
        embedding = [i.item() for i in embedding]

        query['name'] = name
        query['embeddings'] = embedding
        self.collection.insert_one(query)
