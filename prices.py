import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime

def scrape_and_save():
    url = "https://vegetablemarketprice.com/market/maharashtra/today"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://vegetablemarketprice.com/'
    }

    try:
        print("🌐 Connecting to website...")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        print("🔍 Parsing HTML content...")
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='table')

        if not table:
            raise ValueError("Price table not found")

        print("📊 Extracting price data...")
        prices = []
        for row in table.find_all('tr')[1:]:  # skip header
            cols = [td.get_text(strip=True) for td in row.find_all('td')]
            if len(cols) >= 6:
                # Skip first column
                data = cols[1:6]
                prices.append({
                    'vegetable': data[0],
                    'wholesale': data[1],
                    'retail': data[2],
                    'shopping_mall': data[3],
                    'units': data[4]
                })

        print(f"✅ Found {len(prices)} vegetable entries")

        with open('prices.json', 'w', encoding='utf-8') as f:
            json.dump({
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prices': prices
            }, f, indent=2, ensure_ascii=False)

        print("💾 Data saved to prices.json")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        with open('prices.json', 'w') as f:
            json.dump({
                'error': str(e),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f)

if __name__ == '__main__':
    scrape_and_save()
