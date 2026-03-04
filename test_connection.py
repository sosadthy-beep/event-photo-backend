from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)

try:
    client.admin.command('ping')
    print("MongoDB Atlas 連接成功！")
    print("可用 databases：", client.list_database_names())
except Exception as e:
    print("連接失敗：", str(e))