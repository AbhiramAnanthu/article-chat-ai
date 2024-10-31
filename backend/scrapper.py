import requests
from bs4 import BeautifulSoup
from backend.exceptionHandling import ElementNotFoundException
import re


class DataExtractor:
    def scrape(self, url: str):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            try:
                element = soup.select_one("article")
                if element is None:
                    raise ElementNotFoundException()
            except ElementNotFoundException as e:
                element = soup.select_one("body")
            try:
                with open("article.txt", "w", encoding="utf-8") as file:
                    file.write(element.text)
            except IOError as e:
                print(f"Error: {e}")
        except requests.exceptions.HTTPError as e:
            print(f"Error connecting website: {e}")

    def cleaning(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                text.strip()
                lines = text.split("\n")
                newLines = ["".join(line.strip()) for line in lines]
                try:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.writelines(newLines)
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")

    def tokens(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                tokens = re.findall(r"\b\w+\b", text)
                return len(tokens)
        except FileNotFoundError as e:
            print(f"File not found: {e}")

