import requests
from bs4 import BeautifulSoup
import pandas as pd

class Scraper:
    def __init__(self, base_url, total_pages=10, thread_base_url=None):
        self.base_url = base_url
        self.total_pages = total_pages  # Number of forum pages
        self.thread_base_url = thread_base_url or base_url  # Base URL for threads
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        }
        self.data = []

    def scrape(self):
        # Scrape the main forum pages
        for page_number in range(1, self.total_pages + 1):
            if page_number == 1:
                page_url = self.base_url
            else:
                page_url = f"{self.base_url}/page-{page_number}"
            
            response = requests.get(page_url, headers=self.headers)
            
            if response.status_code == 200:
                print(f"Scraping page: {page_url}")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find thread links on the main page with the specific href pattern
                thread_links = soup.find_all('a', href=True)

                for link in thread_links:
                    thread_url = link['href']
                    
                    # Filter only links that match the pattern /community/index.php?threads/
                    if '/community/index.php?threads/' in thread_url:
                        full_thread_url = self.thread_base_url + thread_url

                        # Visit the thread and scrape all pages within the thread
                        self.scrape_thread(full_thread_url)
            else:
                print(f"Failed to retrieve page {page_number}. Status code: {response.status_code}")

    def scrape_thread(self, thread_url):
        """Scrape individual thread and handle pagination."""
        while thread_url:
            print(f"Scraping thread: {thread_url}")
            response = requests.get(thread_url, headers=self.headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Scrape the current page
                self.scrape_thread_page(soup)

                # Check if there's a "Next" page link and get its URL
                next_page_link = soup.find('a', class_='pageNav-jump pageNav-jump--next', href=True)
                
                if next_page_link:
                    next_page_url = next_page_link['href']
                    thread_url = self.thread_base_url + next_page_url  # Update to the next page URL
                else:
                    # No more pages, stop the loop
                    break
            else:
                print(f"Failed to retrieve thread page {thread_url}. Status code: {response.status_code}")
                break

    def scrape_thread_page(self, soup):
        """Extract comments and dates from a single thread page."""
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

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def save_to_csv(self, filename):
        df = self.to_dataframe()
        df.to_csv(filename, index=False)

# Example usage:
scraper = Scraper("https://pokegym.net/community/index.php?forums/vg-news-gossip.148", total_pages=12, thread_base_url="https://pokegym.net")
scraper.scrape()
scraper.save_to_csv("scraped_threads.csv")

print("Data saved to scraped_threads.csv")
