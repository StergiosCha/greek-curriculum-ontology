import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging
from tqdm.asyncio import tqdm
import re
import time

from app.core.config import settings

logger = logging.getLogger(__name__)

class GreekCurriculumScraper:
    """Scrapes curriculum PDFs from ebooks.edu.gr"""
    
    def __init__(self):
        self.base_url = settings.ebooks_base_url
        self.curricula_dir = Path(settings.curricula_dir)
        self.curricula_dir.mkdir(exist_ok=True)
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_curriculum_links(self) -> List[Dict[str, str]]:
        """Extract curriculum PDF links from the ebooks site"""
        logger.info(f"Fetching curriculum links from {self.base_url}")
        
        try:
            async with self.session.get(self.base_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch main page: {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                curriculum_links = []
                
                # Look for links containing curriculum-related terms
                curriculum_patterns = [
                    r'αναλυτικ[οάή].*προγραμμ',
                    r'νέα ελληνικά',
                    r'γλώσσα.*λογοτεχνία',
                    r'δημοτικ[οό]',
                    r'γυμνάσι[οό]',
                    r'λύκει[οό]'
                ]
                
                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    text = link.get_text(strip=True).lower()
                    
                    # Check if link text matches curriculum patterns
                    for pattern in curriculum_patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            # Check if it's a PDF link or leads to PDF
                            if href.endswith('.pdf') or 'pdf' in href.lower():
                                full_url = urljoin(self.base_url, href)
                                curriculum_links.append({
                                    'title': link.get_text(strip=True),
                                    'url': full_url,
                                    'type': self._classify_curriculum_type(text)
                                })
                                break
                
                # Also look for PDF links in iframe or embedded content
                curriculum_links.extend(await self._find_embedded_pdfs(soup))
                
                logger.info(f"Found {len(curriculum_links)} curriculum links")
                return curriculum_links
                
        except Exception as e:
            logger.error(f"Error fetching curriculum links: {e}")
            return []
    
    def _classify_curriculum_type(self, text: str) -> str:
        """Classify curriculum by education level"""
        text = text.lower()
        if any(term in text for term in ['δημοτικ', 'πρωτοβάθμι']):
            return 'primary'
        elif any(term in text for term in ['γυμνάσι', 'δευτεροβάθμι']):
            return 'secondary_lower'
        elif any(term in text for term in ['λύκει', 'λυκειακ']):
            return 'secondary_upper'
        else:
            return 'unknown'
    
    async def _find_embedded_pdfs(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Find PDFs in iframes or other embedded content"""
        embedded_links = []
        
        # Check iframes
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src')
            if src and src.endswith('.pdf'):
                full_url = urljoin(self.base_url, src)
                embedded_links.append({
                    'title': iframe.get('title', 'Embedded Curriculum PDF'),
                    'url': full_url,
                    'type': 'embedded'
                })
        
        # Check object/embed tags
        for obj in soup.find_all(['object', 'embed']):
            data = obj.get('data') or obj.get('src')
            if data and data.endswith('.pdf'):
                full_url = urljoin(self.base_url, data)
                embedded_links.append({
                    'title': obj.get('title', 'Embedded Curriculum PDF'),
                    'url': full_url,
                    'type': 'embedded'
                })
        
        return embedded_links
    
    async def download_curriculum(self, curriculum: Dict[str, str]) -> Optional[str]:
        """Download a single curriculum PDF"""
        try:
            # Create safe filename
            title = curriculum['title']
            safe_title = re.sub(r'[^\w\s-]', '', title)
            safe_title = re.sub(r'[-\s]+', '_', safe_title)
            filename = f"{safe_title}_{curriculum['type']}.pdf"
            
            filepath = self.curricula_dir / filename
            
            # Skip if already exists
            if filepath.exists():
                logger.info(f"Curriculum already exists: {filename}")
                return str(filepath)
            
            logger.info(f"Downloading: {curriculum['title']}")
            
            async with self.session.get(curriculum['url']) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Verify it's actually a PDF
                    if content.startswith(b'%PDF'):
                        async with aiofiles.open(filepath, 'wb') as f:
                            await f.write(content)
                        
                        logger.info(f"Successfully downloaded: {filename}")
                        return str(filepath)
                    else:
                        logger.warning(f"Downloaded content is not a PDF: {curriculum['url']}")
                        return None
                else:
                    logger.error(f"Failed to download {curriculum['url']}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading {curriculum['title']}: {e}")
            return None
    
    async def download_all_curricula(self, curricula: List[Dict[str, str]]) -> List[str]:
        """Download all curriculum PDFs with rate limiting"""
        if not curricula:
            logger.warning("No curricula to download")
            return []
        
        logger.info(f"Starting download of {len(curricula)} curricula")
        
        # Create semaphore for concurrent downloads
        semaphore = asyncio.Semaphore(settings.max_concurrent_downloads)
        
        async def download_with_semaphore(curriculum):
            async with semaphore:
                result = await self.download_curriculum(curriculum)
                # Rate limiting
                await asyncio.sleep(settings.request_delay)
                return result
        
        # Download with progress bar
        tasks = [download_with_semaphore(curr) for curr in curricula]
        results = await tqdm.gather(*tasks, desc="Downloading curricula")
        
        # Filter successful downloads
        successful_downloads = [path for path in results if path is not None]
        
        logger.info(f"Successfully downloaded {len(successful_downloads)} curricula")
        return successful_downloads
    
    async def scrape_and_download(self) -> Dict[str, List[str]]:
        """Main method to scrape and download all curricula"""
        logger.info("Starting curriculum scraping and download")
        
        # Get all curriculum links
        curriculum_links = await self.get_curriculum_links()
        
        if not curriculum_links:
            logger.warning("No curriculum links found")
            return {'downloaded': [], 'failed': []}
        
        # Download all curricula
        downloaded_paths = await self.download_all_curricula(curriculum_links)
        
        failed_count = len(curriculum_links) - len(downloaded_paths)
        
        return {
            'downloaded': downloaded_paths,
            'failed': failed_count,
            'total_found': len(curriculum_links)
        }

# Convenience function for external use
async def scrape_greek_curricula() -> Dict[str, List[str]]:
    """Scrape and download Greek curricula"""
    async with GreekCurriculumScraper() as scraper:
        return await scraper.scrape_and_download()
