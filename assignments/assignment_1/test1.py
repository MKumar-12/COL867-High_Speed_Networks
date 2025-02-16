import os
import json
import time
import argparse
import subprocess
from browsermobproxy import Server
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


# Path to include executables
BROWSERMOB_PROXY_PATH  = "C:\\Utility\\browsermob-proxy-2.1.4\\bin\\browsermob-proxy.bat"
CHROMEDRIVER_PATH = "C:\\Program Files (x86)\\chromedriver.exe"
TSHARK_PATH = "C:\\Program Files\\Wireshark\\tshark.exe"


# Directory for storing intermediate files
INTERMEDIATE_DIR = "intermediate_files"
HAR_FILE = os.path.join(INTERMEDIATE_DIR, "yt_video.har")
PCAP_FILE = os.path.join(INTERMEDIATE_DIR, "yt_video.pcap")


# fn. to clear existing files
def clear_files():
    if os.path.exists(HAR_FILE):
        os.remove(HAR_FILE)
    if os.path.exists(PCAP_FILE):
        os.remove(PCAP_FILE)
    

# fn. to start BrowserMob Proxy
def start_proxy(proxy_path):
    server = Server(proxy_path)
    server.start()
    return server

# fn. to configure Selenium WebDriver
def configure_driver(driver_path, proxy_server):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument(f"--proxy-server={proxy_server}")
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


# fn. to start capturing HAR logs
def start_har_capture(proxy, name="youtube"):
    proxy.new_har(name, options={"captureHeaders": True, "captureContent": True})

# fn. to start packet capture using tshark
def start_packet_capture(tshark_path, pcap_file, interface="Wi-Fi"):
    tshark_cmd = [
        tshark_path,
        "-i", interface, 
        "-w", pcap_file
    ]
    return subprocess.Popen(tshark_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(video_url):
    clear_files()
    proxy_server = start_proxy(BROWSERMOB_PROXY_PATH)
    driver = configure_driver(CHROMEDRIVER_PATH, proxy_server.proxy)

    try:
        start_har_capture(proxy_server)
        tshark_process = start_packet_capture(TSHARK_PATH, PCAP_FILE)

        # Load the YouTube video
        driver.get(video_url)
        print("Video started. Capturing traffic for 3 minutes...")
        print(f"Title: {driver.title}")

        time.sleep(180)  # Allow time for playback and capture network traffic for 3 min.

        # Save HAR file
        print("Stopping capture...")
        har_data = proxy_server.har
        with open(HAR_FILE, "w") as har_file:
            json.dump(har_data, har_file, indent=4)

        # Stop packet capture
        tshark_process.terminate()
        
        print(f"HAR file saved: {HAR_FILE}")
        print(f"PCAP file saved: {PCAP_FILE}")

    finally:
        # Close browser and proxy
        driver.quit()
        proxy_server.stop()
        

# python main.py --url "https://www.youtube.com/watch?v=WevXTsed_oc"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="YouTube video URL")
    
    args = parser.parse_args()
    main(args.url)