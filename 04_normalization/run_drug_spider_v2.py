import pandas as pd
from datetime import datetime, UTC
import scrapy
from scrapy.crawler import CrawlerProcess

# --- CONFIGURATION / SETUP ---

# 1. API_KEY and urlencode are no longer needed, so we remove the imports/function.
# from urllib.parse import urlencode 
# API_KEY is removed/not needed for this free solution.

# --- DATA LOADING ---
# drugbank_from_umls.csv
# drugbank_full_database_with_product_names.csv

drug_ids = pd.read_csv(
    "./04_normalization/data/term_dictionaries/drugbank_full_20251115/drugbank_from_umls.csv"
)["primary_drugbank_id"].tolist()
print(f"Total drug IDs from DrugBank: {len(drug_ids)}")

try:
    drugbank_external_ids = pd.read_json("drugbank_external_ids.jsonl", lines=True)
    scraped_ids = drugbank_external_ids["drugbank_id"].unique().tolist()
except FileNotFoundError:
    print("drugbank_external_ids.jsonl not found. Starting fresh scrape.")
    scraped_ids = []

# Filter out scraped IDs
remaining_drug_ids = [did for did in drug_ids if did not in scraped_ids]
#remaining_drug_ids = remaining_drug_ids[:100]  # Limit to first 100 IDs for testing
print(f"Total drug IDs to scrape: {len(remaining_drug_ids)}")


# --- SPIDER DEFINITION ---

class DrugSpider(scrapy.Spider):
    name = "drug"
    # Update allowed_domains to the actual target site
    allowed_domains = ["go.drugbank.com"] 

    # Removed custom_settings here, they will go into CrawlerProcess

    def start_requests(self):
        """Yields requests for the remaining DrugBank IDs, enabling Playwright."""
        for drug_id in remaining_drug_ids:
            url = f"https://go.drugbank.com/drugs/{drug_id}"
            
            yield scrapy.Request(
                url=url, # Use the direct URL
                callback=self.parse,
                meta={
                    'drug_id': drug_id,
                    # Crucially, tell Scrapy to use Playwright for this request
                    "playwright": True, 
                }
            )

    def parse(self, response):
        drugbank_id = response.meta['drug_id'] 
        
        # FIX 1: Using 'now(UTC)'
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

# --- RUN SPIDER ----

# configure_logging() # configure_logging is often unnecessary when setting LOG_LEVEL
process = CrawlerProcess(settings={
    # --- OUTPUT SETTINGS (Original) ---
    "FEEDS": {
        "drugbank_external_ids.jsonl": {
            "format": "jsonlines",
            "overwrite": False,   # APPEND MODE
            "encoding": "utf8"
        }
    },
    "LOG_LEVEL": "INFO",
    
    # --- PLAYWRIGHT CONFIGURATION (New) ---
    # 1. Enable Playwright Downloader Middleware
    "DOWNLOAD_HANDLERS": {
        "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    },
    
    # 2. Configure Playwright to use Chromium
    "PLAYWRIGHT_BROWSER_TYPE": "chromium", 
    
    # 3. Adjust concurrency (Playwright requests are slower/resource intensive)
    "CONCURRENT_REQUESTS": 4, 
    "CONCURRENT_REQUESTS_PER_DOMAIN": 2, 
    
    # 4. Increase timeout for browser operations
    "DOWNLOAD_TIMEOUT": 60, 
    
    # 5. Optional: Default Scrapy user-agent is sometimes blocked, set a common one
    "USER_AGENT": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
})


process.crawl(DrugSpider)
process.start()