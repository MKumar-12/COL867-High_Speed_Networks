import os
from browsermobproxy import Server
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import json
import time
import subprocess

BROWSERMOB_PROXY_PATH  = "C:\\Utility\\browsermob-proxy-2.1.4\\bin\\browsermob-proxy.bat"
CHROMEDRIVER_PATH = "C:\\Program Files (x86)\\chromedriver.exe"
TShark_path = "C:\\Program Files\\Wireshark\\tshark.exe"

# Create a directory to store the intermediate files
INTERMEDIATE_DIR = "intermediate_files"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

# Path to store the HAR and PCAP files
HAR_file = os.path.join(INTERMEDIATE_DIR, "yt_video.har")
PCAP_file = os.path.join(INTERMEDIATE_DIR, "yt_video.pcap")

# Clear earlier created files (if they exist)
if os.path.exists(HAR_file):
    os.remove(HAR_file)
if os.path.exists(PCAP_file):
    os.remove(PCAP_file)
    
# Start BrowserMob Proxy
server = Server(BROWSERMOB_PROXY_PATH)
server.start()
proxy = server.create_proxy()

# Configure Selenium to use the proxy
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU usage to avoid related issues
chrome_options.add_argument("--ignore-certificate-errors")  # Ignore SSL errors
chrome_options.add_argument(f"--proxy-server={proxy.proxy}")

# Launch the WebDriver with the configured options
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Start capturing HAR logs
proxy.new_har("youtube", options={"captureHeaders": True, "captureContent": True})

# Start packet capture using tshark
tshark_cmd = [
    TShark_path,
    "-i", "Wi-Fi",  # Change to your network interface name (use `tshark -D` to list)
    "-w", PCAP_file
]
tshark_process = subprocess.Popen(tshark_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Load the YouTube video
video_url = "https://www.youtube.com/watch?v=WevXTsed_oc"
driver.get(video_url)
print("Video started. Capturing traffic for 3 minutes...")

time.sleep(180)  # Allow time for playback and capture network traffic

print(driver.title)

# Save HAR file
print("Stopping capture...")
har_data = proxy.har
with open(HAR_file, "w") as har_file:
    json.dump(har_data, har_file, indent=4)

# Stop packet capture
tshark_process.terminate()

# Close browser and proxy
driver.quit()
server.stop()

print(f"HAR file saved: {HAR_file}")
print(f"PCAP file saved: {PCAP_file}")