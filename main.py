import streamlit as st
import random


class Interface:
    def __init__(self, url, article_name, id) -> None:
        self.id = id
        self.url = url
        self.article_name = article_name
        self.collection_name = f"{article_name}_{id}_{random.randint(0,9)}"
        self.index_name = (
            f"{article_name}_{id}_{random.randint(9,16)}_{random.randint(3,8)}"
        )
    
    def checkExists():
