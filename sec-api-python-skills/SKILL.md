---
name: sec-api-python-skills
description: Access and analyze SEC EDGAR filings data using comprehensive APIs for searching, downloading, parsing financial statements, insider trading, institutional holdings, and extracting structured data from 10-K, 10-Q, 8-K and other SEC forms.
---

# SEC API Python Skills

Access the entire SEC EDGAR filings database with powerful search, download, and parsing capabilities for financial analysis and compliance workflows.

## What This Skill Does

This skill provides comprehensive access to SEC EDGAR data through Python:

1. **Filing Search & Retrieval** - Search 18M+ filings by ticker, CIK, form type, date range, and more
2. **Real-Time Stream** - Get new filings as they're published via WebSocket
3. **Full-Text Search** - Search the complete text of all filings since 2001
4. **XBRL-to-JSON** - Extract standardized financial statements from 10-K/10-Q
5. **Section Extraction** - Parse specific sections from 10-K, 10-Q, 8-K filings
6. **Insider Trading** - Access Form 3/4/5 insider buy/sell transactions
7. **Institutional Holdings** - Query 13F holdings and cover pages
8. **Ownership Data** - Track 13D/13G activist investor positions
9. **Investment Advisers** - Search Form ADV filings
10. **Fund Holdings** - Access N-PORT mutual fund/ETF holdings
11. **Executive Data** - Get director, board member, and compensation data
12. **Enforcement** - Search SEC enforcement actions and litigation
13. **CUSIP/CIK/Ticker Mapping** - Resolve identifiers and company details

## Prerequisites

Before using this skill, ensure you have:

1. **Python Package**:
   ```bash
   pip install sec-api
   ```

2. **API Key**: Get a free API key from [sec-api.io](https://sec-api.io)

3. **For Real-Time Stream**: Install WebSockets support
   ```bash
   pip install websockets
   ```

## Core APIs and Examples

### 1. Query API - Search SEC Filings

Search and filter all 18M filings by ticker, CIK, form type, filing date, items, and more.

**Basic Search Example**:
```python
from sec_api import QueryApi

queryApi = QueryApi(api_key="YOUR_API_KEY")

# Get all TSLA 10-Q filings in 2020
query = {
    "query": "ticker:TSLA AND filedAt:[2020-01-01 TO 2020-12-31] AND formType:\"10-Q\"",
    "from": "0",
    "size": "10",
    "sort": [{"filedAt": {"order": "desc"}}]
}

filings = queryApi.get_filings(query)
```

**Find 8-K with Specific Items**:
```python
# Find Form 8-K with Item 9.01 "Financial Statements and Exhibits"
query = {
    "query": "formType:\"8-K\" AND items:\"9.01\"",
    "from": "0",
    "size": "10",
    "sort": [{"filedAt": {"order": "desc"}}]
}

filings = queryApi.get_filings(query)
```

### 2. Full-Text Search API

Search the full text of all filings and their exhibits since 2001.

**Example**:
```python
from sec_api import FullTextSearchApi

fullTextSearchApi = FullTextSearchApi(api_key="YOUR_API_KEY")

query = {
    "query": '"artificial intelligence"',
    "formTypes": ['10-K', '10-Q'],
    "startDate": '2023-01-01',
    "endDate": '2023-12-31',
}

filings = fullTextSearchApi.get_filings(query)
```

### 3. Filing & Exhibit Download API

Download any SEC filing or exhibit in its original format.

**Example**:
```python
from sec_api import RenderApi

renderApi = RenderApi(api_key="YOUR_API_KEY")

# Download HTML filing
url_8k = "https://www.sec.gov/Archives/edgar/data/1045810/000104581023000014/nvda-20230222.htm"
filing = renderApi.get_file(url_8k)

# Download binary files (PDF, Excel, images)
url_pdf = "https://www.sec.gov/Archives/edgar/data/1798925/999999999724004095/filename1.pdf"
pdf_file = renderApi.get_file(url_pdf, return_binary=True)

# Save to disk
with open("filing.pdf", "wb") as f:
    f.write(pdf_file)
```

### 4. XBRL-to-JSON Converter API

Extract standardized financial statements from any 10-K or 10-Q.

**Example**:
```python
from sec_api import XbrlApi

xbrlApi = XbrlApi("YOUR_API_KEY")

# Convert using filing URL
xbrl_json = xbrlApi.xbrl_to_json(
    htm_url="https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/aapl-20200926.htm"
)

# Access financial statements
income_statement = xbrl_json["StatementsOfIncome"]
balance_sheet = xbrl_json["BalanceSheets"]
cash_flow = xbrl_json["StatementsOfCashFlows"]

# Get revenue from income statement
revenue = income_statement["RevenueFromContractWithCustomerExcludingAssessedTax"]
```

**Available Statements**:
- `StatementsOfIncome` / `StatementsOfIncomeParenthetical`
- `StatementsOfComprehensiveIncome` / `StatementsOfComprehensiveIncomeParenthetical`
- `BalanceSheets` / `BalanceSheetsParenthetical`
- `StatementsOfCashFlows` / `StatementsOfCashFlowsParenthetical`
- `StatementsOfShareholdersEquity` / `StatementsOfShareholdersEquityParenthetical`
- `CoverPage` (company metadata)

### 5. Section Extractor API

Extract specific sections from 10-K, 10-Q, and 8-K filings.

**10-K Sections**:
```python
from sec_api import ExtractorApi

extractorApi = ExtractorApi("YOUR_API_KEY")

filing_url = "https://www.sec.gov/Archives/edgar/data/1318605/000156459021004599/tsla-10k_20201231.htm"

# Extract section 1A "Risk Factors" as text
risk_factors = extractorApi.get_section(filing_url, "1A", "text")

# Extract section 7 "MD&A" as HTML
mda = extractorApi.get_section(filing_url, "7", "html")
```

**Available 10-K Sections**: 1, 1A, 1B, 1C, 2, 3, 4, 5, 6, 7, 7A, 8, 9, 9A, 9B, 10, 11, 12, 13, 14

**10-Q Sections**: Part1Item1, Part1Item2, Part1Item3, Part1Item4, Part2Item1, Part2Item1A, Part2Item2, Part2Item5, Part2Item6

**8-K Items**: 1-1, 1-2, 2-1, 2-2, 4-1, 4-2, 5-2, 7-1, 8-1, 9-1, etc.

### 6. Real-Time Stream API

Get live stream of newly published filings via WebSocket.

**Example**:
```python
import asyncio
import websockets
import json

API_KEY = "YOUR_API_KEY"
WS_ENDPOINT = f"wss://stream.sec-api.io?apiKey={API_KEY}"

async def websocket_client():
    async with websockets.connect(WS_ENDPOINT) as websocket:
        print("Connected to SEC filing stream")

        while True:
            message = await websocket.recv()
            filings = json.loads(message)

            for filing in filings:
                print(f"{filing['ticker']} - {filing['formType']} - {filing['filedAt']}")

asyncio.run(websocket_client())
```

### 7. Insider Trading API (Form 3/4/5)

Access all insider buy/sell transactions.

**Example**:
```python
from sec_api import InsiderTradingApi

insiderTradingApi = InsiderTradingApi("YOUR_API_KEY")

# Get insider trades for TSLA
trades = insiderTradingApi.get_data({
    "query": "issuer.tradingSymbol:TSLA",
    "from": "0",
    "size": "50",
    "sort": [{"filedAt": {"order": "desc"}}]
})

for transaction in trades["transactions"]:
    owner = transaction["reportingOwner"]["name"]
    shares = transaction["nonDerivativeTable"]["transactions"][0]["amounts"]["shares"]
    print(f"{owner} traded {shares} shares")
```

### 8. Form 13F Institutional Holdings

Access institutional portfolio holdings.

**Example**:
```python
from sec_api import Form13FHoldingsApi

form13FApi = Form13FHoldingsApi(api_key="YOUR_API_KEY")

# Get holdings for a specific fund
query = {
    "query": "cik:1350694 AND periodOfReport:2024-03-31",
    "from": "0",
    "size": "100",
    "sort": [{"filedAt": {"order": "desc"}}]
}

response = form13FApi.get_data(query)
holdings = response["data"]

for holding in holdings:
    print(f"{holding['nameOfIssuer']}: {holding['value']} shares")
```

### 9. Form 13D/13G Activist Investor API

Track activist and passive investor positions (5%+ ownership).

**Example**:
```python
from sec_api import Form13DGApi

form13DGApi = Form13DGApi("YOUR_API_KEY")

# Find filings disclosing 10%+ ownership
query = {
    "query": "owners.amountAsPercent:[10 TO *]",
    "from": "0",
    "size": "50",
    "sort": [{"filedAt": {"order": "desc"}}]
}

filings = form13DGApi.get_data(query)

for filing in filings["filings"]:
    print(f"{filing['nameOfIssuer']}: {filing['owners'][0]['name']} owns {filing['owners'][0]['amountAsPercent']}%")
```

### 10. CUSIP/CIK/Ticker Mapping API

Resolve company identifiers and get company details.

**Example**:
```python
from sec_api import MappingApi

mappingApi = MappingApi(api_key="YOUR_API_KEY")

# Resolve by different identifiers
by_ticker = mappingApi.resolve("ticker", "TSLA")
by_cik = mappingApi.resolve("cik", "1318605")
by_cusip = mappingApi.resolve("cusip", "88160R101")

# Get all companies on an exchange
nasdaq_companies = mappingApi.resolve("exchange", "NASDAQ")

# Company object includes:
# - name, ticker, cik, cusip, exchange
# - sector, industry, sic, sicSector, sicIndustry
# - location, currency, isDelisted
```

### 11. Executive Compensation API

Get standardized compensation data from DEF 14A filings.

**Example**:
```python
from sec_api import ExecCompApi

execCompApi = ExecCompApi("YOUR_API_KEY")

# Get compensation by ticker
result = execCompApi.get_data("TSLA")

# Query specific criteria
query = {
    "query": "cik:70858 AND (year:2023 OR year:2024)",
    "from": "0",
    "size": "50",
    "sort": [{"year": {"order": "desc"}}, {"total": {"order": "desc"}}]
}

compensation = execCompApi.get_data(query)

for exec in compensation:
    print(f"{exec['name']}: ${exec['total']:,.0f} ({exec['year']})")
```

### 12. Directors & Board Members API

Access all directors and board members data.

**Example**:
```python
from sec_api import DirectorsBoardMembersApi

directorsApi = DirectorsBoardMembersApi("YOUR_API_KEY")

query = {
    "query": "ticker:AMZN",
    "from": "0",
    "size": "50",
    "sort": [{"filedAt": {"order": "desc"}}]
}

directors = directorsApi.get_data(query)
```

### 13. Form ADV Investment Advisers API

Search Form ADV filings by advisory firms and individuals.

**Example**:
```python
from sec_api import FormAdvApi

formAdvApi = FormAdvApi("YOUR_API_KEY")

# Search by CRD number
firms = formAdvApi.get_firms({
    "query": "Info.FirmCrdNb:361",
    "from": "0",
    "size": "10"
})

# Get ownership structures
direct_owners = formAdvApi.get_direct_owners(crd="793")
indirect_owners = formAdvApi.get_indirect_owners(crd="326262")
private_funds = formAdvApi.get_private_funds(crd="793")
```

## Common Workflow Patterns

### Pattern 1: Track Company Filings

```python
from sec_api import QueryApi

queryApi = QueryApi(api_key="YOUR_API_KEY")

def get_company_filings(ticker, form_types, start_date, end_date):
    """Get all filings for a company within a date range"""
    form_query = " OR ".join([f'formType:"{ft}"' for ft in form_types])

    query = {
        "query": f"ticker:{ticker} AND ({form_query}) AND filedAt:[{start_date} TO {end_date}]",
        "from": "0",
        "size": "100",
        "sort": [{"filedAt": {"order": "desc"}}]
    }

    return queryApi.get_filings(query)

# Get all 10-K and 10-Q filings for AAPL in 2023
filings = get_company_filings("AAPL", ["10-K", "10-Q"], "2023-01-01", "2023-12-31")
```

### Pattern 2: Extract Financial Statements

```python
from sec_api import QueryApi, XbrlApi

queryApi = QueryApi(api_key="YOUR_API_KEY")
xbrlApi = XbrlApi("YOUR_API_KEY")

def get_latest_financials(ticker):
    """Get the most recent 10-K financial statements"""
    # Find latest 10-K
    query = {
        "query": f"ticker:{ticker} AND formType:\"10-K\"",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}]
    }

    filings = queryApi.get_filings(query)
    latest = filings["filings"][0]

    # Extract XBRL data
    xbrl_json = xbrlApi.xbrl_to_json(accession_no=latest["accessionNo"])

    return {
        "company": latest["companyName"],
        "filedAt": latest["filedAt"],
        "income_statement": xbrl_json.get("StatementsOfIncome"),
        "balance_sheet": xbrl_json.get("BalanceSheets"),
        "cash_flow": xbrl_json.get("StatementsOfCashFlows")
    }

financials = get_latest_financials("TSLA")
```

### Pattern 3: Monitor Insider Trading

```python
from sec_api import InsiderTradingApi

insiderApi = InsiderTradingApi("YOUR_API_KEY")

def monitor_insider_buying(ticker, min_value=100000):
    """Find significant insider purchases"""
    trades = insiderApi.get_data({
        "query": f"issuer.tradingSymbol:{ticker}",
        "from": "0",
        "size": "50",
        "sort": [{"filedAt": {"order": "desc"}}]
    })

    buys = []
    for transaction in trades["transactions"]:
        for t in transaction["nonDerivativeTable"]["transactions"]:
            if t["coding"]["code"] == "P":  # Purchase
                value = t["amounts"]["shares"] * t["amounts"]["pricePerShare"]
                if value >= min_value:
                    buys.append({
                        "insider": transaction["reportingOwner"]["name"],
                        "shares": t["amounts"]["shares"],
                        "price": t["amounts"]["pricePerShare"],
                        "value": value,
                        "date": transaction["periodOfReport"]
                    })

    return buys

significant_buys = monitor_insider_buying("AAPL", min_value=500000)
```

### Pattern 4: Track Institutional Ownership Changes

```python
from sec_api import Form13FHoldingsApi

form13FApi = Form13FHoldingsApi(api_key="YOUR_API_KEY")

def track_institutional_changes(cusip, periods):
    """Compare institutional holdings across quarters"""
    holdings_by_period = {}

    for period in periods:
        query = {
            "query": f"holdings.cusip:{cusip} AND periodOfReport:{period}",
            "from": "0",
            "size": "1000"
        }

        response = form13FApi.get_data(query)
        holdings_by_period[period] = response["data"]

    return holdings_by_period

# Compare Q4 2023 vs Q1 2024
changes = track_institutional_changes("88160R101", ["2023-12-31", "2024-03-31"])
```

### Pattern 5: Extract Risk Factors

```python
from sec_api import QueryApi, ExtractorApi

queryApi = QueryApi(api_key="YOUR_API_KEY")
extractorApi = ExtractorApi("YOUR_API_KEY")

def get_risk_factors(ticker):
    """Extract risk factors from latest 10-K"""
    # Find latest 10-K
    query = {
        "query": f"ticker:{ticker} AND formType:\"10-K\"",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}]
    }

    filings = queryApi.get_filings(query)
    filing_url = filings["filings"][0]["linkToFilingDetails"]

    # Extract section 1A
    risk_factors = extractorApi.get_section(filing_url, "1A", "text")

    return risk_factors

risks = get_risk_factors("TSLA")
```

## Best Practices

### 1. API Key Management
```python
import os

# Use environment variables
api_key = os.getenv("SEC_API_KEY")
queryApi = QueryApi(api_key=api_key)
```

### 2. Efficient Querying
- **Use specific queries**: Narrow down with ticker, date range, form type
- **Pagination**: Use `from` and `size` parameters for large result sets
- **Sort strategically**: Sort by `filedAt` for chronological analysis

```python
# Good: Specific query
query = "ticker:AAPL AND formType:\"10-K\" AND filedAt:[2023-01-01 TO 2023-12-31]"

# Bad: Too broad
query = "formType:\"10-K\""
```

### 3. Error Handling
```python
try:
    filings = queryApi.get_filings(query)
except Exception as e:
    print(f"Query failed: {e}")
    # Implement retry logic or fallback
```

### 4. Rate Limiting
- Free tier: Rate limits apply
- Implement exponential backoff for retries
- Cache results when possible

### 5. Proxy Support
```python
proxies = {
    "http": "http://your-proxy.com",
    "https": "https://your-proxy.com"
}

queryApi = QueryApi(api_key="YOUR_API_KEY", proxies=proxies)
```

## Advanced Use Cases

### 1. ESG Disclosure Tracking
```python
def find_esg_disclosures(ticker, keywords):
    """Find filings mentioning specific ESG topics"""
    fullTextApi = FullTextSearchApi(api_key="YOUR_API_KEY")

    query = {
        "query": f'"{keywords}"',
        "ticker": ticker,
        "formTypes": ['10-K', '10-Q', 'DEF 14A'],
        "startDate": '2020-01-01'
    }

    return fullTextApi.get_filings(query)

climate_disclosures = find_esg_disclosures("AAPL", "climate change")
```

### 2. M&A Activity Monitoring
```python
def monitor_ma_activity(start_date):
    """Monitor M&A announcements via 8-K Item 2.01"""
    queryApi = QueryApi(api_key="YOUR_API_KEY")

    query = {
        "query": f'formType:"8-K" AND items:"2.01" AND filedAt:[{start_date} TO *]',
        "from": "0",
        "size": "100",
        "sort": [{"filedAt": {"order": "desc"}}]
    }

    return queryApi.get_filings(query)
```

### 3. Auditor Change Detection
```python
def find_auditor_changes(year):
    """Find companies changing auditors (Item 4.01)"""
    item_api = Form_8K_Item_X_Api("YOUR_API_KEY")

    query = {
        "query": f"item4_01:* AND filedAt:[{year}-01-01 TO {year}-12-31]",
        "from": "0",
        "size": "100",
        "sort": [{"filedAt": {"order": "desc"}}]
    }

    return item_api.get_data(query)
```

### 4. Fund Holdings Analysis
```python
def analyze_fund_positions(cik, period):
    """Analyze a fund's top positions"""
    form13FApi = Form13FHoldingsApi(api_key="YOUR_API_KEY")

    query = {
        "query": f"cik:{cik} AND periodOfReport:{period}",
        "from": "0",
        "size": "1000",
        "sort": [{"value": {"order": "desc"}}]
    }

    holdings = form13FApi.get_data(query)

    # Calculate concentration
    total_value = sum(h["value"] for h in holdings["data"])
    top_10_value = sum(h["value"] for h in holdings["data"][:10])

    return {
        "total_positions": len(holdings["data"]),
        "total_value": total_value,
        "top_10_concentration": top_10_value / total_value
    }
```

## Troubleshooting

### API Key Issues
**Problem**: Authentication failed

**Solution**:
1. Verify API key is valid at sec-api.io
2. Check key is correctly passed to API class
3. Ensure no extra spaces or quotes in key

### Empty Results
**Problem**: Query returns no results

**Solution**:
1. Verify ticker/CIK is correct
2. Check date format: `YYYY-MM-DD`
3. Test with broader query first
4. Check form type spelling (case-sensitive)

### XBRL Parsing Errors
**Problem**: Cannot extract financial statements

**Solution**:
1. Verify filing has XBRL data (check `linkToXbrl`)
2. Some older filings may not have XBRL
3. Try different URL formats (htm_url, xbrl_url, accession_no)

### WebSocket Connection Issues
**Problem**: Real-time stream disconnects

**Solution**:
1. Implement reconnection logic
2. Check network/firewall settings
3. Verify API key has stream access

## Performance Optimization

### Caching Results
```python
import json
from pathlib import Path

def cached_query(query, cache_file):
    """Cache query results to disk"""
    cache_path = Path(cache_file)

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    results = queryApi.get_filings(query)

    with open(cache_path, 'w') as f:
        json.dump(results, f)

    return results
```

### Batch Processing
```python
def batch_download_filings(urls, batch_size=10):
    """Download filings in batches"""
    renderApi = RenderApi(api_key="YOUR_API_KEY")

    results = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i+batch_size]
        for url in batch:
            results.append(renderApi.get_file(url))
        time.sleep(1)  # Rate limiting

    return results
```

## Resources

- **Documentation**: [sec-api.io/docs](https://sec-api.io/docs)
- **API Portal**: [sec-api.io](https://sec-api.io)
- **Supported Form Types**: [sec-api.io/list-of-sec-filing-types](https://sec-api.io/list-of-sec-filing-types)
- **PyPI Package**: [pypi.org/project/sec-api](https://pypi.org/project/sec-api)
- **GitHub**: Search for "sec-api-python" examples

## When to Use This Skill

Use this skill when you need to:
- Search and download SEC filings programmatically
- Extract financial statements from 10-K/10-Q reports
- Monitor real-time filing publications
- Track insider trading activities
- Analyze institutional ownership changes
- Research activist investor positions
- Parse specific sections from regulatory filings
- Build financial compliance workflows
- Aggregate data across multiple companies
- Perform quantitative financial analysis
- Monitor corporate events (M&A, auditor changes, etc.)
- Extract executive compensation data
- Track investment adviser activities
- Research enforcement actions and litigation

This skill is essential for financial analysts, quantitative researchers, compliance teams, data scientists, and developers building fintech applications that require comprehensive SEC data access.
