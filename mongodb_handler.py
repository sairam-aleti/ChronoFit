"""
MongoDB Handler for ChronoFit - Handles all database operations for feedback storage and retrieval
"""

import os
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import pandas as pd
from datetime import datetime

class MongoDBHandler:
    def __init__(self):
        self.client = None
        self.db = None
        self.feedback_collection = None
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            # Get connection string from Streamlit secrets or environment
            try:
                mongo_uri = st.secrets.get("MONGODB_URI", "") or os.getenv("MONGODB_URI", "")
            except:
                # If Streamlit context not available, use environment variable only
                mongo_uri = os.getenv("MONGODB_URI", "")
            
            if not mongo_uri:
                raise ValueError("MONGODB_URI not found in secrets or environment")
            
            # Set connection timeout to 5 seconds
            self.client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                retryWrites=False
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Set database and collection
            self.db = self.client['chronofit']
            self.feedback_collection = self.db['user_feedback']
            
            # Create index on timestamp for efficient queries
            self.feedback_collection.create_index('timestamp')
            
        except (ConnectionFailure, ServerSelectionTimeoutError, ValueError) as e:
            # Silent fail - don't interrupt app
            self.client = None
            self.db = None
            self.feedback_collection = None
        except Exception as e:
            # Catch any other exceptions
            self.client = None
            self.db = None
            self.feedback_collection = None
            return False
        
        return True
    
    def is_connected(self):
        """Check if MongoDB is connected"""
        return self.client is not None and self.db is not None
    
    def save_feedback(self, feedback_record):
        """
        Save feedback record to MongoDB
        
        Args:
            feedback_record (dict): Feedback data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in feedback_record:
                feedback_record['timestamp'] = datetime.utcnow()
            
            result = self.feedback_collection.insert_one(feedback_record)
            return bool(result.inserted_id)
        except Exception as e:
            return False
    
    def get_all_feedback(self):
        """
        Retrieve all feedback records from MongoDB
        
        Returns:
            pd.DataFrame: All feedback records or empty DataFrame if error
        """
        if not self.is_connected():
            return pd.DataFrame()
        
        try:
            records = list(self.feedback_collection.find({}, {'_id': 0}))
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Convert timestamp to datetime if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def get_feedback_count(self):
        """
        Get total count of feedback records
        
        Returns:
            int: Number of feedback records
        """
        if not self.is_connected():
            return 0
        
        try:
            return self.feedback_collection.count_documents({})
        except Exception as e:
            return 0
    
    def get_last_retrain_count(self):
        """
        Get the last recorded feedback count at retraining
        
        Returns:
            int: Last retrain feedback count
        """
        if not self.is_connected():
            return 0
        
        try:
            metadata_collection = self.db['metadata']
            record = metadata_collection.find_one({'_id': 'last_retrain'})
            return record.get('feedback_count', 0) if record else 0
        except Exception as e:
            return 0
    
    def update_last_retrain_count(self, count):
        """
        Update the last recorded feedback count after retraining
        
        Args:
            count (int): Current feedback count
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        try:
            metadata_collection = self.db['metadata']
            metadata_collection.update_one(
                {'_id': 'last_retrain'},
                {
                    '$set': {
                        'feedback_count': count,
                        'last_updated': datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()


# Global handler instance
_handler = None

def get_mongodb_handler():
    """Get or create MongoDB handler instance"""
    global _handler
    if _handler is None:
        _handler = MongoDBHandler()
    return _handler
