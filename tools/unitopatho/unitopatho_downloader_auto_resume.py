import os
import sys
import subprocess
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime, timedelta

class UniToPathoDownloader:
    def __init__(self):
        self.cookies = {
            'simplesamlphp_auth_returnto': 'https://ieee-dataport.org/open-access/unitopatho',
            'SimpleSAMLSessionID': '2a11ddcd56c70d40e9230b440b6cfd65',
            'SimpleSAMLAuthToken': '_d24b54793067e2b7a3c25683b11d08c4bae6c23c82',
            'SSESSf80f5dcc65a99d0290714d0bd9d60a3a': 'orRcU2D0Xy1U1pFo693iW0TjOajMkeo4SWXXd3GuxcDssgkA'
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.base_url = "https://ieee-dataport.org"
        self.dataset_url = f"{self.base_url}/open-access/unitopatho"
        self.output_path = "UNITOPatho.zip"
        self.chunk_size = 1024 * 1024 * 2
        self.current_url = None
        self.url_timestamp = None
        self.url_expiry_minutes = 9

    def test_download(self, url):
        try:
            response = requests.get(url, cookies=self.cookies, headers=self.headers, stream=True, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                content_length = response.headers.get('content-length')
                content_type = response.headers.get('content-type', '')
                content_disposition = response.headers.get('content-disposition', '')
                if content_length:
                    size_gb = int(content_length) / (1024**3)
                    if size_gb > 200:
                        return True
                    elif size_gb < 1:
                        return False
                chunk = next(response.iter_content(chunk_size=8192))
                if chunk:
                    if b'<!DOCTYPE' in chunk or b'<html' in chunk:
                        return False
                    if content_length and int(content_length) > 200 * (1024**3):
                        return True
                    elif 'zip' in content_type or 'zip' in content_disposition:
                        return True
            return False
        except:
            return False
        finally:
            try:
                response.close()
            except:
                pass

    def find_download_link(self):
        print("Fetching fresh download URL...")
        response = requests.get(self.dataset_url, cookies=self.cookies, headers=self.headers)
        if response.status_code != 200:
            print(f"Failed to fetch page (status: {response.status_code})")
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        all_links = soup.find_all('a', href=True)
        download_candidates = []
        for link in all_links:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            if any(keyword in href.lower() for keyword in ['download', '.zip', 'unitopath']):
                full_url = href if href.startswith('http') else self.base_url + (href if href.startswith('/') else '/' + href)
                download_candidates.append((full_url, text))
        download_divs = soup.find_all('div', class_=lambda x: x and 'download' in x.lower() if x else False)
        for div in download_divs:
            links = div.find_all('a', href=True)
            for link in links:
                href = link.get('href', '')
                full_url = href if href.startswith('http') else self.base_url + (href if href.startswith('/') else '/' + href)
                if full_url not in [c[0] for c in download_candidates]:
                    download_candidates.append((full_url, link.get_text(strip=True)))
        for url, text in download_candidates:
            if self.test_download(url):
                print("Found working download URL")
                self.current_url = url
                self.url_timestamp = datetime.now()
                return url
        return None

    def is_url_expired(self):
        if not self.url_timestamp:
            return True
        return datetime.now() - self.url_timestamp > timedelta(minutes=self.url_expiry_minutes)

    def get_fresh_url_if_needed(self):
        if self.is_url_expired() or not self.current_url:
            return self.find_download_link()
        return self.current_url

    def download_chunk(self, start_byte=0):
        url = self.get_fresh_url_if_needed()
        if not url:
            return None, None, None
        headers = self.headers.copy()
        if start_byte > 0:
            headers['Range'] = f'bytes={start_byte}-'
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            if response.status_code not in [200, 206]:
                if response.status_code == 403:
                    self.current_url = None
                    self.url_timestamp = None
                    return None, None, None
                elif response.status_code == 416:
                    return response, 0, start_byte
                return None, None, None
            total_size = int(response.headers.get('content-length', 0))
            if 'content-range' in response.headers:
                total_size = int(response.headers['content-range'].split('/')[-1])
            elif start_byte > 0:
                total_size += start_byte
            return response, total_size, start_byte
        except:
            return None, None, None

    def download(self):
        resume_byte_pos = 0
        if os.path.exists(self.output_path):
            resume_byte_pos = os.path.getsize(self.output_path)
            print(f"Resuming from: {resume_byte_pos / (1024**3):.2f} GB")
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            response, total_size, current_pos = self.download_chunk(resume_byte_pos)
            if response is None:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = min(30 * retry_count, 120)
                    print(f"Retry {retry_count}/{max_retries} in {wait_time}s...")
                    time.sleep(wait_time)
                continue
            if total_size == current_pos:
                print("Download complete!")
                return True
            retry_count = 0
            mode = 'ab' if resume_byte_pos > 0 else 'wb'
            try:
                with open(self.output_path, mode) as f:
                    if total_size > 0:
                        print(f"Total size: {total_size / (1024**3):.2f} GB")
                        with tqdm(total=total_size, initial=resume_byte_pos, unit='B', unit_scale=True, desc="Downloading", ascii=True) as pbar:
                            start_time = time.time()
                            downloaded_session = 0
                            last_refresh_check = time.time()
                            for chunk in response.iter_content(chunk_size=self.chunk_size):
                                if chunk:
                                    if time.time() - last_refresh_check > 60:
                                        if self.is_url_expired():
                                            print("URL expiring soon, will refresh on next retry")
                                            response.close()
                                            resume_byte_pos = os.path.getsize(self.output_path)
                                            break
                                        last_refresh_check = time.time()
                                    f.write(chunk)
                                    chunk_size_bytes = len(chunk)
                                    pbar.update(chunk_size_bytes)
                                    downloaded_session += chunk_size_bytes
                                    resume_byte_pos += chunk_size_bytes
                                    elapsed = time.time() - start_time
                                    if elapsed > 0:
                                        speed = downloaded_session / elapsed
                                        pbar.set_postfix(speed=f"{speed/(1024**2):.1f}MB/s")
                    else:
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                resume_byte_pos += len(chunk)
                                print(f"Downloaded: {resume_byte_pos / (1024**3):.2f} GB", end='\r')
                response.close()
                if os.path.getsize(self.output_path) == total_size and total_size > 0:
                    print("Download complete!")
                    return True
            except KeyboardInterrupt:
                print("Download interrupted, run again to resume")
                return False
            except Exception as e:
                print(f"Error: {e}")
                resume_byte_pos = os.path.getsize(self.output_path) if os.path.exists(self.output_path) else 0
        print("Max retries reached")
        return False

def setup_environment():
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)
    pip_path = os.path.join(venv_dir, 'bin', 'pip')
    python_path = os.path.join(venv_dir, 'bin', 'python')
    subprocess.run([pip_path, 'install', 'requests', 'tqdm', 'beautifulsoup4'], check=True)
    subprocess.run([python_path, __file__, '--skip-setup'] + sys.argv[1:], check=False)
    sys.exit(0)

if __name__ == "__main__":
    if '--skip-setup' not in sys.argv:
        setup_environment()
    print("UniToPatho Dataset Auto-Downloader")
    print("=" * 50)
    downloader = UniToPathoDownloader()
    if downloader.download():
        print("✓ Download successful!")
    else:
        print("Download failed")