from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime 
from sqlalchemy.ext.declarative import declarative_base 
import datetime 

Base = declarative_base() 

class TextChunk(Base): 
 __tablename__ = "text_chunks" 
 id = Column(Integer, primary_key=True, autoincrement=True)  
 text = Column(Text) 
 is_annotated = Column(Boolean, default=False) 
 annotation_json = Column(Text, nullable=True) 
 timestamp = Column(DateTime, default=datetime.datetime.utcnow) 

class MolecularTarget(Base): 
 __tablename__ = "molecular_targets" 
 id = Column(Integer, primary_key=True, autoincrement=True)  
 name = Column(String) 
 my = Column(Float) 
 mn = Column(Float) 
 hesitancy = Column(Float) 
 confidence = Column(Float) 
 timestamp = Column(DateTime, default=datetime.datetime.utcnow)
