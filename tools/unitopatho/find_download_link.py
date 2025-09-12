import requests
from bs4 import BeautifulSoup

def test_download(url, cookies):
    print(f"Testing download from: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, cookies=cookies, headers=headers, stream=True, timeout=10, allow_redirects=True)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            content_length = response.headers.get('content-length')
            content_type = response.headers.get('content-type', '')
            content_disposition = response.headers.get('content-disposition', '')
            
            if content_length:
                size_gb = int(content_length) / (1024**3)
                print(f"File size: {size_gb:.2f} GB")
                
                if size_gb > 200:
                    print("This looks like the right file (>200GB)!")
                elif size_gb < 1:
                    print("File too small, not the dataset")
                    return False
            
            chunk = next(response.iter_content(chunk_size=8192))
            if chunk:
                if chunk[:2] == b'PK':
                    print("File appears to be a ZIP archive")
                elif b'<!DOCTYPE' in chunk or b'<html' in chunk:
                    print("This is an HTML page, not the dataset")
                    return False
                
                if content_length and int(content_length) > 200 * (1024**3):
                    print("Download is working! This is the dataset!")
                    return True
                elif 'zip' in content_type or 'zip' in content_disposition:
                    print("Download is working!")
                    return True
                else:
                    print("Download works but might not be the right file")
                    return False
        else:
            print(f"Failed with status code: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("Connection error")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        try:
            response.close()
        except:
            pass

def find_download_link():
    cookies = {
    'simplesamlphp_auth_returnto': 'https://ieee-dataport.org/open-access/unitopatho',
    'SimpleSAMLSessionID': '2a11ddcd56c70d40e9230b440b6cfd65',
    'SimpleSAMLAuthToken': '_d24b54793067e2b7a3c25683b11d08c4bae6c23c82',
    'SSESSf80f5dcc65a99d0290714d0bd9d60a3a': 'orRcU2D0Xy1U1pFo693iW0TjOajMkeo4SWXXd3GuxcDssgkA'
    }
    
    base_url = "https://ieee-dataport.org"
    dataset_url = f"{base_url}/open-access/unitopatho"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    print(f"Fetching: {dataset_url}")
    
    response = requests.get(dataset_url, cookies=cookies, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: Failed to fetch page (status: {response.status_code})")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    print("Searching for download links...")
    
    all_links = soup.find_all('a', href=True)
    download_candidates = []
    
    for link in all_links:
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        if any(keyword in href.lower() for keyword in ['download', '.zip', 'unitopath']):
            full_url = href if href.startswith('http') else base_url + (href if href.startswith('/') else '/' + href)
            download_candidates.append((full_url, text))
            print(f"Found candidate: {full_url} | Text: '{text}'")
    
    download_divs = soup.find_all('div', class_=lambda x: x and 'download' in x.lower() if x else False)
    for div in download_divs:
        links = div.find_all('a', href=True)
        for link in links:
            href = link.get('href', '')
            full_url = href if href.startswith('http') else base_url + (href if href.startswith('/') else '/' + href)
            if full_url not in [c[0] for c in download_candidates]:
                download_candidates.append((full_url, link.get_text(strip=True)))
                print(f"Found in download div: {full_url}")
    
    print(f"Total candidates found: {len(download_candidates)}")
    
    if download_candidates:
        print("Testing download links...")
        
        for url, text in download_candidates:
            if test_download(url, cookies):
                print(f"WORKING DOWNLOAD LINK FOUND:")
                print(f"  {url}")
                return url
        
        print("None of the links worked for download")
    else:
        print("No download links found")
        print("Showing first 10 links on page for debugging:")
        for i, link in enumerate(all_links[:10]):
            print(f"{i+1}. {link.get('href', 'NO HREF')[:100]}")

if __name__ == "__main__":
    find_download_link()