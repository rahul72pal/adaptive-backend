import logging
from typing import Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

logger = logging.getLogger(__name__)

class InMemoryMongoFallback:
    """Lightweight in-memory document store used if live MongoDB is unavailable."""
    def __init__(self):
        self._collections: Dict[str, Dict[str, Any]] = {
            "concept_graphs": {},
            "learning_paths": {},
            "questions": {}
        }

    class CollectionWrapper:
        def __init__(self, store: Dict[str, Any]):
            self._store = store

        async def find_one(self, filter_dict: dict) -> Optional[dict]:
            for doc in self._store.values():
                match = True
                for k, v in filter_dict.items():
                    if doc.get(k) != v:
                        match = False
                        break
                if match:
                    return doc
            return None

        async def insert_one(self, doc: dict):
            doc_id = doc.get("_id") or doc.get("id") or str(len(self._store) + 1)
            doc["_id"] = doc_id
            self._store[str(doc_id)] = doc
            return type("InsertResult", (), {"inserted_id": doc_id})()

        async def update_one(self, filter_dict: dict, update_dict: dict, upsert: bool = False):
            existing = await self.find_one(filter_dict)
            if existing:
                if "$set" in update_dict:
                    existing.update(update_dict["$set"])
                return existing
            elif upsert:
                new_doc = filter_dict.copy()
                if "$set" in update_dict:
                    new_doc.update(update_dict["$set"])
                return await self.insert_one(new_doc)
            return None

        def find(self, filter_dict: dict):
            results = []
            for doc in self._store.values():
                match = True
                for k, v in filter_dict.items():
                    if doc.get(k) != v:
                        match = False
                        break
                if match:
                    results.append(doc)

            class AsyncCursor:
                def __init__(self, data):
                    self._data = data

                async def to_list(self, length: int = None):
                    return self._data[:length] if length else self._data

            return AsyncCursor(results)

        async def count_documents(self, filter_dict: dict) -> int:
            count = 0
            for doc in self._store.values():
                match = True
                for k, v in filter_dict.items():
                    if doc.get(k) != v:
                        match = False
                        break
                if match:
                    count += 1
            return count

    def get_collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = {}
        return InMemoryMongoFallback.CollectionWrapper(self._collections[name])

class ResilientMongoDatabase:
    """Wraps live Motor DB and automatically falls back to InMemoryMongoFallback on network timeout."""
    def __init__(self, real_db):
        self._real_db = real_db
        self._fallback = InMemoryMongoFallback()
        self._use_fallback = False

    class ResilientCollection:
        def __init__(self, real_coll, fallback_coll, parent):
            self._real_coll = real_coll
            self._fallback_coll = fallback_coll
            self._parent = parent

        async def find_one(self, filter_dict: dict):
            if self._parent._use_fallback:
                return await self._fallback_coll.find_one(filter_dict)
            try:
                return await self._real_coll.find_one(filter_dict)
            except Exception as e:
                logger.warning(f"MongoDB query failed ({e}). Switching to in-memory fallback...")
                self._parent._use_fallback = True
                return await self._fallback_coll.find_one(filter_dict)

        async def insert_one(self, doc: dict):
            if self._parent._use_fallback:
                return await self._fallback_coll.insert_one(doc)
            try:
                return await self._real_coll.insert_one(doc)
            except Exception as e:
                logger.warning(f"MongoDB insert failed ({e}). Switching to in-memory fallback...")
                self._parent._use_fallback = True
                return await self._fallback_coll.insert_one(doc)

        async def update_one(self, filter_dict: dict, update_dict: dict, upsert: bool = False):
            if self._parent._use_fallback:
                return await self._fallback_coll.update_one(filter_dict, update_dict, upsert)
            try:
                return await self._real_coll.update_one(filter_dict, update_dict, upsert)
            except Exception as e:
                logger.warning(f"MongoDB update failed ({e}). Switching to in-memory fallback...")
                self._parent._use_fallback = True
                return await self._fallback_coll.update_one(filter_dict, update_dict, upsert)

        def find(self, filter_dict: dict):
            if self._parent._use_fallback:
                return self._fallback_coll.find(filter_dict)
            try:
                cursor = self._real_coll.find(filter_dict)
                # Wrap Motor cursor to catch to_list timeouts
                class ResilientCursor:
                    def __init__(self, real_cursor, fallback_cursor, parent):
                        self._real_cursor = real_cursor
                        self._fallback_cursor = fallback_cursor
                        self._parent = parent

                    async def to_list(self, length: int = None):
                        if self._parent._use_fallback:
                            return await self._fallback_cursor.to_list(length)
                        try:
                            return await self._real_cursor.to_list(length)
                        except Exception as e:
                            logger.warning(f"MongoDB cursor fetch failed ({e}). Switching to in-memory fallback...")
                            self._parent._use_fallback = True
                            return await self._fallback_cursor.to_list(length)

                return ResilientCursor(cursor, self._fallback_coll.find(filter_dict), self._parent)
            except Exception as e:
                logger.warning(f"MongoDB find failed ({e}). Switching to in-memory fallback...")
                self._parent._use_fallback = True
                return self._fallback_coll.find(filter_dict)

        async def count_documents(self, filter_dict: dict) -> int:
            if self._parent._use_fallback:
                return await self._fallback_coll.count_documents(filter_dict)
            try:
                return await self._real_coll.count_documents(filter_dict)
            except Exception as e:
                logger.warning(f"MongoDB count failed ({e}). Switching to in-memory fallback...")
                self._parent._use_fallback = True
                return await self._fallback_coll.count_documents(filter_dict)

    def get_collection(self, name: str):
        if self._use_fallback:
            return self._fallback.get_collection(name)
        real_coll = self._real_db.get_collection(name)
        fallback_coll = self._fallback.get_collection(name)
        return ResilientMongoDatabase.ResilientCollection(real_coll, fallback_coll, self)

mongo_client: Optional[AsyncIOMotorClient] = None
mongo_db = None
is_in_memory_fallback = False

async def init_mongo() -> None:
    """Initializes MongoDB connection or activates in-memory fallback."""
    global mongo_client, mongo_db, is_in_memory_fallback
    tls_kwargs = {}
    try:
        import certifi
        tls_kwargs["tlsCAFile"] = certifi.where()
    except Exception:
        pass

    try:
        mongo_client = AsyncIOMotorClient(
            settings.MONGO_URL,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
            socketTimeoutMS=2000,
            **tls_kwargs
        )
        # Ping the server
        await mongo_client.admin.command('ping')
        real_db = mongo_client[settings.MONGO_DB_NAME]
        mongo_db = ResilientMongoDatabase(real_db)
        logger.info("Successfully connected to MongoDB.")
    except Exception as e:
        logger.warning(f"MongoDB connection failed ({e}). Activating in-memory Mongo fallback store...")
        is_in_memory_fallback = True
        mongo_db = InMemoryMongoFallback()

def get_mongo_db():
    """Dependency for providing MongoDB database handle."""
    global mongo_db
    if mongo_db is None:
        mongo_db = InMemoryMongoFallback()
    return mongo_db
