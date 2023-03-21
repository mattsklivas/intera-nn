import os
from os import environ as env
from dotenv import find_dotenv, load_dotenv
from pymongo import MongoClient
import json
from bson import json_util

# load the environment variables from the .env file
load_dotenv(find_dotenv())

basedir = os.path.abspath(os.path.dirname(__file__))

class Database(object):
    client = MongoClient(f"mongodb+srv://{env.get('USERNAME')}:{env.get('PASSWORD')}@{env.get('DATABASE_URL')}")
    intera_calls_db = 'intera_calls'
    rooms_collection = 'rooms'
    messages_collection = 'messages'

def parse_json(data):
    return json.loads(json_util.dumps(data))
