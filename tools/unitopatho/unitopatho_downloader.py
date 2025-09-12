import os
import sys
import subprocess
import time

def setup_environment():
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)
    
    pip_path = os.path.join(venv_dir, 'bin', 'pip')
    python_path = os.path.join(venv_dir, 'bin', 'python')
    
    subprocess.run([pip_path, 'install', 'requests', 'tqdm'], check=True)
    subprocess.run([python_path, __file__, '--skip-setup'] + sys.argv[1:], check=False)
    sys.exit(0)

if '--skip-setup' not in sys.argv:
    setup_environment()

import requests
from tqdm import tqdm

def download_with_resume(url, output_path, chunk_size=1024*1024*2):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    if os.path.exists(output_path):
        resume_byte_pos = os.path.getsize(output_path)
        print(f"Resuming download from: {resume_byte_pos / (1024**3):.2f} GB")
        headers['Range'] = f'bytes={resume_byte_pos}-'
    else:
        resume_byte_pos = 0
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        if response.status_code not in [200, 206]:
            print(f"Error: Server returned status code {response.status_code}")
            if response.status_code == 403:
                print("URL expired! Get a fresh URL by running find_download_link.py")
            elif response.status_code == 416:
                print("File already completely downloaded")
                return True
            return False
        
        if response.status_code != 206:
            resume_byte_pos = 0
        
        total_size = int(response.headers.get('content-length', 0))
        if 'content-range' in response.headers:
            total_size = int(response.headers['content-range'].split('/')[-1])
        elif resume_byte_pos > 0:
            total_size += resume_byte_pos
        
        if total_size == 0:
            mode = 'ab' if resume_byte_pos > 0 else 'wb'
            with open(output_path, mode) as f:
                downloaded = resume_byte_pos
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        print(f"Downloaded: {downloaded / (1024**3):.2f} GB", end='\r')
        else:
            print(f"Total file size: {total_size / (1024**3):.2f} GB")
            
            mode = 'ab' if resume_byte_pos > 0 else 'wb'
            with open(output_path, mode) as f:
                with tqdm(total=total_size, initial=resume_byte_pos, unit='B', 
                         unit_scale=True, desc="Downloading", ascii=True) as pbar:
                    start_time = time.time()
                    downloaded_session = 0
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            chunk_size_bytes = len(chunk)
                            pbar.update(chunk_size_bytes)
                            downloaded_session += chunk_size_bytes
                            
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                speed = downloaded_session / elapsed
                                pbar.set_postfix(speed=f"{speed/(1024**2):.1f}MB/s")
        
        print(f"Download completed: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return False
    except KeyboardInterrupt:
        print("Download interrupted by user")
        print("Run the script again to resume")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    print("UniToPatho Dataset Downloader")
    print("=" * 50)
    
    download_url = "https://ieee-dataport.s3.amazonaws.com/open/49617/UNITOPatho.zip?versionId=Ea17jXGkYwfwgIGxauamyLrFLr3eAHt7&response-content-disposition=attachment%3B%20filename%3D%22UNITOPatho.zip%22&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20250909%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250909T184129Z&X-Amz-SignedHeaders=host&X-Amz-Expires=600&X-Amz-Signature=5260bb9a54429d3caeaac921ea628092382682903cc3618ee632fb14303f7b68"
    
    if len(sys.argv) > 1 and sys.argv[-1].startswith("http"):
        download_url = sys.argv[-1]
    else:
        print("Using hardcoded URL (expires in 10 minutes)")
        print("To use a fresh URL, run: python unitopatho_downloader_final.py <URL>")
    
    output_filename = "UNITOPatho.zip"
    
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        if retry_count > 0:
            print(f"Retry attempt {retry_count} of {max_retries}")
        
        if download_with_resume(download_url, output_filename):
            print("✓ Download successful!")
            break
        else:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = min(30 * retry_count, 120)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Maximum retries reached")
                print("The URL may have expired. Get a fresh URL by running find_download_link.py")
                sys.exit(1)

if __name__ == "__main__":
    main()