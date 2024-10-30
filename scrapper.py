import requests
from bs4 import BeautifulSoup
import re

url = "https://iamtrask.github.io/2015/07/12/basic-python-network/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
element = soup.select_one("article")
with open("article.txt","w",encoding="utf-8") as file:
    file.write(element.text)