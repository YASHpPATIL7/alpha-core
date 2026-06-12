import requests, time, pandas as pd
s = requests.Session()
s.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
s.get('https://www.nseindia.com', timeout=15)  # get cookie
time.sleep(2)
url = 'https://www.nseindia.com/api/fiidiiTradeReact?type=fiiDii&from=01-01-2019&to=07-06-2026'
r = s.get(url, timeout=20)
df = pd.DataFrame(r.json())
print(df.head())
df.to_csv('fii_raw.csv', index=False)