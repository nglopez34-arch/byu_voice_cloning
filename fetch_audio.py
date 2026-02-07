import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import re


# Change right here and nowhere else!! (If you are following instructions.txt)
# Paste the full URL for the author you want to download
# Example: https://speeches.byu.edu/speakers/jeffrey-r-holland/
author_page = "https://speeches.byu.edu/speakers/jeffrey-r-holland/"

# Extract author slug from URL
author_slug = author_page.rstrip('/').split('/')[-1]

# Create downloads directory
downloads_dir = Path(f"{author_slug.replace('-', '_')}_speeches")
downloads_dir.mkdir(exist_ok=True)

# Base URL for BYU speeches
base_url = "https://speeches.byu.edu"

print(f"Fetching list of speeches for {author_slug}...")
response = requests.get(author_page)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all speech links
speech_links = []
for link in soup.find_all('a', href=True):
    href = link['href']
    if f'/talks/{author_slug}/' in href and href not in speech_links:
        full_url = href if href.startswith('http') else base_url + href
        speech_links.append(full_url)

print(f"Found {len(speech_links)} speeches")

# Download MP3 for each speech
for i, speech_url in enumerate(speech_links, 1):
    try:
        print(f"\n[{i}/{len(speech_links)}] Processing: {speech_url}")

        # Get the speech page
        speech_response = requests.get(speech_url)
        speech_soup = BeautifulSoup(speech_response.text, 'html.parser')

        # Find MP3 download link
        mp3_link = None
        for link in speech_soup.find_all('a', href=True):
            if link['href'].endswith('.mp3'):
                mp3_link = link['href']
                break

        if not mp3_link:
            print(f"  ⚠ No MP3 found")
            continue

        # Make sure we have full URL
        if not mp3_link.startswith('http'):
            mp3_link = base_url + mp3_link

        # Extract filename from URL or create from speech title
        title = speech_soup.find('h1')
        if title:
            filename = re.sub(r'[^\w\s-]', '', title.text.strip())
            filename = re.sub(r'[-\s]+', '_', filename)
            filename = f"{filename}.mp3"
        else:
            filename = mp3_link.split('/')[-1]

        filepath = downloads_dir / filename

        # Skip if already downloaded
        if filepath.exists():
            print(f"  ✓ Already exists: {filename}")
            continue

        # Download the MP3
        print(f"  ⬇ Downloading: {filename}")
        mp3_response = requests.get(mp3_link, stream=True)
        mp3_response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in mp3_response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  ✓ Saved: {filename}")

        # Be nice to the server
        time.sleep(1)

    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

print(f"\n{'=' * 50}")
print(f"Download complete! Files saved to: {downloads_dir.absolute()}")
