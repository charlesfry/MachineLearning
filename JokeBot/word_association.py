from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import re

outpath = 'E:\WordAssociations'

def single_associations(word:str) :
    """
    gets all word associations with a single word
    :param word: word to be checked
    :return:
    """
    base_url = 'https://wordassociations.net/en/words-associated-with/'
    url = base_url + word

    page = requests.get(url)
    assert str(page.status_code).startswith('2'), f'Page failed to load. Error code: {page.status_code}'

    soup = BeautifulSoup(page.content, 'html.parser')
    print(soup)

def compare_associations(first:str, second:str):
    pass


if __name__ == '__main__':
    single_associations('butt')