import requests
from bs4 import BeautifulSoup
import pandas as pd

class Scraper:
    def __init__(self, base_url, total_pages=10):
        self.base_url = base_url
        self.total_pages = total_pages
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        }
        self.data = []

    def scrape(self):
        for page_number in range(1, self.total_pages + 1):
            if page_number == 1:
                page_url = self.base_url
            else:
                page_url = f"{self.base_url}/page-{page_number}"
            
            response = requests.get(page_url, headers=self.headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('div', class_='message-cell message-cell--main')
                
                for article in articles:
                    comment_div = article.find('div', class_='bbWrapper')
                    if comment_div:
                        for i_tag in comment_div.find_all('i'):
                            i_tag.decompose()
                        for blockquote in comment_div.find_all('blockquote'):
                            blockquote.decompose()
                        comment_text = comment_div.get_text(strip=True)
                    else:
                        comment_text = 'No comment found'
                    
                    time_tag = article.find('time', class_='u-dt')
                    if time_tag:
                        date_string = time_tag.get('datetime')
                    else:
                        date_string = 'No date found'
                    
                    self.data.append({'date': date_string, 'comment': comment_text})

            else:
                print(f"Failed to retrieve page {page_number}. Status code: {response.status_code}")

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def save_to_csv(self, filename):
        df = self.to_dataframe()
        df.to_csv(filename, index=False)

scrapers = [
    Scraper("https://forums.serebii.net/threads/pokemon-scarlet-violet-general-discussion-thread.761996", total_pages=27),
    Scraper("https://forums.serebii.net/threads/what-things-surprised-you-from-pokemon-scarlet-violet.760160", total_pages=3),
    Scraper("https://forums.serebii.net/threads/most-annoying-things-in-sv-that-arent-glitches.759645", total_pages=11),
    Scraper("https://www.smogon.com/forums/threads/unpopular-opinions-scarlet-violet-edition.3714615", total_pages=7)
]

combined_data = pd.DataFrame()

for scraper in scrapers:
    scraper.scrape()
    combined_data = pd.concat([combined_data, scraper.to_dataframe()], ignore_index=True)

combined_data.to_csv("./data/combined_data.csv", index=False)

print("Data saved to combined_data.csv")
