import os
import sys
import asyncio
import aiohttp
import aiofiles
import json
from pathlib import Path
from tqdm.asyncio import tqdm
import re
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

with open('cookies.json', 'r') as f:
    COOKIES = json.load(f)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'DNT': '1',
    'Referer': 'https://ieee-dataport.org/open-access/unitopatho'
}

class UniToPathoDownloader:
    
    def __init__(self, output_dir: str = "./unitopatho_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = None
        self.semaphore = asyncio.Semaphore(3)
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=3600, connect=60)
        connector = aiohttp.TCPConnector(limit=10, force_close=True)
        self.session = aiohttp.ClientSession(
            headers=HEADERS,
            cookies=COOKIES,
            timeout=timeout,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def verify_auth(self) -> bool:
        async with self.session.get("https://ieee-dataport.org/open-access/unitopatho") as response:
            text = await response.text()
            return 'LOGIN TO ACCESS' not in text
    
    async def fetch_page(self, url: str) -> Optional[str]:
        async with self.semaphore:
            async with self.session.get(url) as response:
                return await response.text() if response.status == 200 else None
    
    async def extract_download_links(self, html: str) -> List[str]:
        patterns = [
            r'href="(/documents/[^"]*download[^"]*)"',
            r'href="([^"]*unitopatho[^"]*\.zip[^"]*)"',
            r'href="(/download/[^"]*)"',
            r'data-url="([^"]*download[^"]*)"',
        ]
        
        links = set()
        for pattern in patterns:
            for match in re.findall(pattern, html, re.IGNORECASE):
                if match.startswith('/'):
                    links.add(f"https://ieee-dataport.org{match}")
                elif match.startswith('http'):
                    links.add(match)
        
        return list(links)
    
    async def probe_download_urls(self) -> List[str]:
        dataset_ids = ["9fsv-tm25", "unitopatho", "28753"]
        url_templates = [
            "https://ieee-dataport.org/download/{id}",
            "https://ieee-dataport.org/documents/{id}/download",
            "https://ieee-dataport.org/datasets/{id}/download",
        ]
        
        probe_urls = [
            template.format(id=dataset_id) 
            for dataset_id in dataset_ids 
            for template in url_templates
        ]
        
        tasks = [self.test_download_url(url) for url in probe_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [url for url, valid in zip(probe_urls, results) if valid is True]
    
    async def test_download_url(self, url: str) -> bool:
        async with self.semaphore:
            async with self.session.head(url, allow_redirects=True) as response:
                if response.status != 200:
                    return False
                
                headers = response.headers
                content_type = headers.get('content-type', '').lower()
                content_disp = headers.get('content-disposition', '').lower()
                
                return (
                    any(x in content_type for x in ['zip', 'octet-stream']) or
                    ('filename=' in content_disp and '.zip' in content_disp)
                )
    
    async def download_file(self, url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
        async with self.semaphore:
            async with self.session.get(url, allow_redirects=True) as response:
                if response.status != 200:
                    return False
                
                total_size = int(response.headers.get('content-length', 0))
                
                content_disp = response.headers.get('content-disposition', '')
                if 'filename=' in content_disp:
                    filename = content_disp.split('filename=')[-1].strip('"\'')
                    dest_path = dest_path.parent / filename
                
                with tqdm(total=total_size, unit='iB', unit_scale=True, 
                         desc=dest_path.name) as progress_bar:
                    async with aiofiles.open(dest_path, 'wb') as file:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            await file.write(chunk)
                            progress_bar.update(len(chunk))
                
                logger.info(f"Downloaded: {dest_path.name} ({total_size / 1024**3:.2f} GB)")
                return True
    
    async def parallel_download(self, urls: List[str]) -> List[Path]:
        tasks = []
        for i, url in enumerate(urls):
            dest_path = self.output_dir / f"unitopatho_part{i+1}.zip"
            tasks.append(self.download_file(url, dest_path))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            self.output_dir / f"unitopatho_part{i+1}.zip"
            for i, result in enumerate(results) 
            if result is True
        ]
    
    async def run(self):
        logger.info("UniToPatho Dataset Downloader")
        logger.info(f"Output: {self.output_dir.absolute()}")
        
        if not await self.verify_auth():
            logger.error("Authentication failed! Please update cookies.")
            return False
        
        logger.info("Authentication successful!")
        
        html = await self.fetch_page("https://ieee-dataport.org/open-access/unitopatho")
        
        download_urls = []
        if html:
            download_urls.extend(await self.extract_download_links(html))
        
        download_urls.extend(await self.probe_download_urls())
        download_urls = list(set(download_urls))
        
        if not download_urls:
            logger.warning("No download URLs found automatically.")
            manual_url = input("Paste download URL (or Enter to exit): ").strip()
            if not manual_url:
                return False
            download_urls = [manual_url]
        
        logger.info(f"Downloading {len(download_urls)} file(s)...")
        downloaded_files = await self.parallel_download(download_urls)
        
        if downloaded_files:
            logger.info(f"Downloaded {len(downloaded_files)} file(s) successfully")
            return True
        else:
            logger.error("No files downloaded")
            return False

async def main():
    async with UniToPathoDownloader() as downloader:
        success = await downloader.run()
        return 0 if success else 1

if __name__ == "__main__":    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)