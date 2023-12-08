import time
import pandas as pd
from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract, extract_metadata

def get_urls_from_sitemap(sitemap_url: str) -> list:
    # Get sitemap and urls
    urls = sitemap_search(sitemap_url)
    return urls


def create_dataset(list_of_websites: list) -> pd.DataFrame:
    data = []

    for website in tqdm(list_of_websites, desc = "Websites"):
        urls = get_urls_from_sitemap(website)
        for url in tqdm(urls, desc="URLs"):
            print("Processing URL:", url)
            html = fetch_url(url)
            body = extract(html)

            try:
                metadata = extract_metadata(html)
                title = metadata.title
                description = metadata.description
            except:
                metadata = ""
                title = ""
                description = ""
            
            d = {
                'url': url,
                'title': title,
                'description': description,
                'body': body
            }
            data.append(d)
            time.sleep(0.5)

    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    df = df.dropna()

    return df


if __name__ == "__main__":
    list_of_websites = [
        "https://python.langchain.com/"
    ]
    df = create_dataset(list_of_websites)
    df.to_csv("./data/dataset.csv", index=False)
