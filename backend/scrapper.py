import requests
from bs4 import BeautifulSoup
from .exceptionHandling import ElementNotFoundException
import re
import tempfile
import os


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
                with open("tempfile.txt", "w", encoding="utf-8") as temp_file:
                    temp_file.write(element.text)
                    temp_path = "tempfile.txt"
                    return temp_path
            except IOError as e:
                print(f"Error: {e}")
        except requests.exceptions.HTTPError as e:
            print(f"Error connecting website: {e}")

    def cleaning(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                text = text.strip()
                lines = text.split("\n")
                newLines = ["".join(line.strip()) for line in lines]
                try:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.writelines(newLines)
                    return file_path
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")


class ScrapeInterface(DataExtractor):
    def __init__(self, url) -> None:
        super().__init__()
        self.url = url

    def scrape_and_clean(self):
        scraped_content = super().scrape(self.url)
        cleaned_scraped_content = super().cleaning(scraped_content)
        with open(cleaned_scraped_content, "r", encoding="utf-8") as file:
            content = file.read()
        return content
