import scrapy
from datetime import datetime, UTC  # <--- 1. Import UTC
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
from urllib.parse import urlencode
import pandas as pd

# --- CONFIGURATION ---
API_KEY = 'c4772c2e-82ff-428e-9cf3-ce2e02cda4f5' 

def get_scrapeops_url(url):
    payload = {'api_key': API_KEY, 'url': url, 'bypass': 'cloudflare_level_2'}
    proxy_url = 'https://proxy.scrapeops.io/v1/?' + urlencode(payload)
    return proxy_url

drug_ids = pd.read_csv(
    "./04_normalization/data/term_dictionaries/drugbank_full_20251115/drugbank_full_database_with_product_names.csv"
)["primary_drugbank_id"].tolist()
print(f"Total drug IDs from DrugBank: {len(drug_ids)}")
drugbank_external_ids = pd.read_json("drugbank_external_ids.jsonl", lines=True)
scraped_ids = drugbank_external_ids["drugbank_id"].unique().tolist()

# Filter out scraped IDs
remaining_drug_ids = [did for did in drug_ids if did not in scraped_ids]

print(f"Total drug IDs to scrape: {len(remaining_drug_ids)}")

class DrugSpider(scrapy.Spider):
    name = "drug"
    allowed_domains = ["proxy.scrapeops.io"]

    custom_settings = {
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    }

    # FIX 2: Renamed 'start_requests' to 'start' and made it 'async'
    async def start(self):
        """THIS is where you create your own requests explicitly."""
        for drug_id in remaining_drug_ids:
            url = f"https://go.drugbank.com/drugs/{drug_id}"
            
            yield scrapy.Request(
                url=get_scrapeops_url(url), 
                callback=self.parse,
                meta={'drug_id': drug_id}
            )

    def parse(self, response):
        drugbank_id = response.meta['drug_id'] 
        
        # FIX 1: Replaced deprecated 'utcnow()' with 'now(UTC)'
        scraped_at_time = datetime.now(UTC) 

        yield {"drugbank_id": drugbank_id, "scraped_at": scraped_at_time}

        ext_ids_section = response.xpath('//dt[@id="external-ids"]')
        if ext_ids_section:
            ul = ext_ids_section.xpath("following-sibling::dd/ul")
            for li in ul.xpath("./li"):
                value = li.xpath("normalize-space(.)").get() 
                if value:
                    yield {
                        "drugbank_id": drugbank_id,
                        "type": "external_id",
                        "value": value,
                        "scraped_at": scraped_at_time,
                    }

# ---- RUN SPIDER ----
configure_logging()
process = CrawlerProcess(settings={
    "FEEDS": {
        "drugbank_external_ids.jsonl": {
            "format": "jsonlines",
            "overwrite": False,   # APPEND MODE
            "encoding": "utf8"
        }
    },
    "LOG_LEVEL": "INFO",
})


process.crawl(DrugSpider)
process.start()