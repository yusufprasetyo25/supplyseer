from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import aiohttp
import sys
import asyncio
from asyncio.proactor_events import _ProactorBasePipeTransport
import warnings
import urllib.parse
import json
import logging
from bs4 import BeautifulSoup
import functools

logger = logging.getLogger(__name__)

def silence_event_loop_closed(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if str(e) != 'Event loop is closed':
                raise
    return wrapper

# Patch ProactorBasePipeTransport
_ProactorBasePipeTransport.__del__ = silence_event_loop_closed(_ProactorBasePipeTransport.__del__)

class GDELTOutputFormat(str, Enum):
    HTML = "html"
    JSON = "json"
    CSV = "csv"

class GDELTMode(str, Enum):
    ARTLIST = "artlist"  # Basic article list
    TIMELINE_VOL = "TimelineVol"  # Volume of coverage timeline
    TIMELINE_TONE = "TimelineTone"  # Emotional timeline
    TIMELINE_VOL_INFO = "TimelineVolInfo"  # Timeline with article details

@dataclass
class GDELTSearchConfig:
    """Configuration for GDELT search parameters"""
    timespan: str = "1d"  # Default to last 24 hours
    format: GDELTOutputFormat = GDELTOutputFormat.JSON
    mode: GDELTMode = GDELTMode.ARTLIST
    max_records: int = 250
    sort: str = "DateDesc"  # Changed from HybridRel to more reliable sort

class SupplyChainGDELTMonitor:
    """GDELT monitor specialized for supply chain risk monitoring"""
    
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self._query_templates = self._initialize_query_templates()

    async def __aenter__(self):
        """Async context manager entry"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
        
    def _initialize_query_templates(self) -> Dict[str, str]:
        """Initialize query templates for different risk types"""
        return {
            "supply_disruption": 'sourcelang:eng (theme:SUPPLY_CHAIN OR theme:PROTEST OR theme:STRIKE) AND tone<-3',
            
            "geopolitical": 'sourcelang:eng (theme:MILITARY_CONFLICT OR theme:TERROR OR theme:TRADE_SANCTIONS)',
            
            "infrastructure": 'sourcelang:eng (theme:TRANSPORT OR theme:PORT OR theme:INFRASTRUCTURE) "disruption closure"',
            
            "trade_restrictions": 'sourcelang:eng (theme:TRADE_SANCTIONS OR theme:EXPORT_CONTROL)',
            
            "natural_disasters": 'sourcelang:eng (theme:NATURAL_DISASTER OR theme:DISASTER) AND tone<-2'
        }

    async def _init_session(self):
        """Initialize aiohttp session if not exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    def _build_url(self, query: str, config: GDELTSearchConfig) -> str:
        """Build GDELT API URL with parameters"""
        params = {
            'query': query,
            'format': config.format.value,
            'mode': config.mode.value,
            'timespan': config.timespan,
            'maxrecords': str(config.max_records),
            'sort': config.sort
        }
        
        query_string = urllib.parse.urlencode(params)
        return f"{self.BASE_URL}?{query_string}"

    async def _fetch_data(self, url: str, config: GDELTSearchConfig) -> Dict:
        """Fetch data from GDELT API with improved error handling"""
        await self._init_session()
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '')
                
                if 'application/json' in content_type:
                    return await response.json()
                elif 'text/html' in content_type:
                    # Handle HTML response
                    html_content = await response.text()
                    return self._parse_html_response(html_content, config.mode)
                else:
                    logger.warning(f"Unexpected content type: {content_type}")
                    return {}
                    
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching data from GDELT: {str(e)}, URL: {url}")
            return {}

    def _parse_html_response(self, html_content: str, mode: GDELTMode) -> Dict:
        """Parse HTML response from GDELT"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        if mode == GDELTMode.ARTLIST:
            articles = []
            for article in soup.find_all('div', class_='article'):
                processed = {
                    'title': article.find('a').text if article.find('a') else None,
                    'url': article.find('a')['href'] if article.find('a') else None,
                    'date': article.find('span', class_='date').text if article.find('span', class_='date') else None
                }
                articles.append(processed)
            return {'articles': articles}
            
        elif mode in [GDELTMode.TIMELINE_VOL, GDELTMode.TIMELINE_TONE]:
            # Extract timeline data from HTML
            # This is a simplified example - adjust based on actual HTML structure
            data_div = soup.find('div', class_='timeline-data')
            if data_div and data_div.get('data-timeline'):
                try:
                    return json.loads(data_div['data-timeline'])
                except json.JSONDecodeError:
                    return {'timeline': []}
            
        return {}

    async def monitor_region_risks(
        self, 
        region: str, 
        timespan: str = "1d"
    ) -> Dict[str, List[Dict]]:
        """Monitor various risk types for a specific region"""
        results = {}
        
        config = GDELTSearchConfig(
            timespan=timespan,
            format=GDELTOutputFormat.JSON,
            mode=GDELTMode.ARTLIST
        )
        
        for risk_type, template in self._query_templates.items():
            # Add region-specific filtering
            region_query = f'{template} AND "{region}"'
            url = self._build_url(region_query, config)
            
            data = await self._fetch_data(url, config)
            results[risk_type] = self._process_articles(data)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        return results

    async def analyze_supply_chain_sentiment(
        self, 
        route: List[str], 
        timespan: str = "7d"
    ) -> Dict[str, Dict]:
        """Analyze sentiment along a supply chain route"""
        config = GDELTSearchConfig(
            timespan=timespan,
            format=GDELTOutputFormat.JSON,
            mode=GDELTMode.TIMELINE_TONE
        )
        
        results = {}
        for location in route:
            query = f'sourcelang:eng (theme:SUPPLY_CHAIN OR theme:TRANSPORT) "{location}"'
            url = self._build_url(query, config)
            
            data = await self._fetch_data(url, config)
            results[location] = self._process_sentiment(data)
            
            await asyncio.sleep(1)
        
        return results

    async def monitor_trade_restrictions(
        self, 
        countries: List[str], 
        commodities: List[str]
    ) -> List[Dict]:
        """Monitor trade restrictions for specific countries and commodities"""
        # Simplify the query structure
        country_terms = ' OR '.join([f'"{country}"' for country in countries])
        commodity_terms = ' OR '.join([f'"{commodity}"' for commodity in commodities])
        
        query = f'sourcelang:eng (theme:TRADE_SANCTIONS OR theme:EXPORT_CONTROL) AND ({country_terms})'
        if commodity_terms:
            query += f' AND ({commodity_terms})'
        
        config = GDELTSearchConfig(
            timespan="30d",
            format=GDELTOutputFormat.JSON,
            mode=GDELTMode.ARTLIST  # Changed from TimelineVolInfo to more reliable ARTLIST
        )
        
        url = self._build_url(query, config)
        data = await self._fetch_data(url, config)
        
        return self._process_trade_restrictions(data)

    def _process_articles(self, data: Dict) -> List[Dict]:
        """Process article data from GDELT response"""
        articles = []
        for article in data.get('articles', []):
            if isinstance(article, dict):  # Handle JSON response
                processed = {
                    'title': article.get('title'),
                    'url': article.get('url'),
                    'source_country': article.get('sourcecountry'),
                    'tone': float(article.get('tone', 0)),
                    'date': article.get('seendate'),
                    'themes': article.get('themes', [])
                }
            else:  # Handle processed HTML response
                processed = article
            articles.append(processed)
        return articles

    def _process_sentiment(self, data: Dict) -> Dict:
        """Process sentiment timeline data"""
        if not data:
            return {
                'average_tone': 0,
                'timeline': [],
                'peak_negative': None,
                'peak_positive': None
            }
            
        timeline = data.get('timeline', [])
        if not timeline:
            return {
                'average_tone': 0,
                'timeline': [],
                'peak_negative': None,
                'peak_positive': None
            }
            
        return {
            'average_tone': sum(t.get('tone', 0) for t in timeline) / len(timeline),
            'timeline': timeline,
            'peak_negative': min(timeline, key=lambda x: x.get('tone', 0), default=None),
            'peak_positive': max(timeline, key=lambda x: x.get('tone', 0), default=None)
        }

    def _process_trade_restrictions(self, data: Dict) -> List[Dict]:
        """Process trade restriction data"""
        return self._process_articles(data)

    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if sys.platform == 'win32':
        # Set event loop policy for Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        async with SupplyChainGDELTMonitor() as monitor:
            try:
                # Monitor risks for a specific region
                risks = await monitor.monitor_region_risks("Red Sea", timespan="7d")
                print("Regional Risks:", json.dumps(risks, indent=2))
                
                # Analyze sentiment along a route
                route_sentiment = await monitor.analyze_supply_chain_sentiment(
                    route=["Singapore", "Suez Canal", "Rotterdam"],
                    timespan="30d"
                )
                print("Route Sentiment:", json.dumps(route_sentiment, indent=2))
                
                # Monitor trade restrictions
                restrictions = await monitor.monitor_trade_restrictions(
                    countries=["China", "USA", "EU"],
                    commodities=["semiconductors", "rare earth metals"]
                )
                print("Trade Restrictions:", json.dumps(restrictions, indent=2))
                    
            except Exception as e:
                logger.error(f"Error in main: {str(e)}")
    finally:
        # Clean up the loop
        loop.stop()
        loop.close()

def run_async_main():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    run_async_main()