import os
import time
import json
import argparse
import logging
import pandas as pd
import pyshark
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from browsermobproxy import Server
import threading

# Configure logging
logging.basicConfig(
    filename="debug_logs.txt",  
    filemode="w",  # "w" to overwrite the file each time, use "a" to append
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_and_print(message, level="info"):
    """Logs the message and prints it to the terminal."""
    print(f"[{level}] {message}")  # Print to terminal
    logging.info(message)
    

# Start BrowserMob Proxy
BROWSERMOB_PROXY_PATH = "/home/manish-kumar/browsermob-proxy/bin/browsermob-proxy"


log_and_print("Starting BrowserMob Proxy...")
server = Server(BROWSERMOB_PROXY_PATH, options={'port': 9090})
server.start()
proxy = server.create_proxy(params={"trustAllServers": "true"})
log_and_print("BrowserMob Proxy started.")


# Function to capture network stats using JavaScript
JS_SCRIPT = """
return {
    resolution: document.querySelector('video') ? document.querySelector('video').videoWidth + 'x' + document.querySelector('video').videoHeight : 'N/A',
    buffer: document.querySelector('video') ? document.querySelector('video').buffered.length > 0 ? document.querySelector('video').buffered.end(0) - document.querySelector('video').currentTime : 0 : 0
};
"""


def setup_webdriver():
    """Configures and launches the Selenium WebDriver."""
    log_and_print("Setting up Selenium WebDriver.")
    chrome_options = Options()
    chrome_options.add_argument(f"--proxy-server={proxy.proxy}")
    # chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--allow-insecure-localhost")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-site-isolation-trials")
    chrome_options.add_argument("--start-maximized")

    service = Service("/usr/bin/chromedriver")  
    driver = webdriver.Chrome(service=service, options=chrome_options)
    log_and_print("Selenium WebDriver setup complete.")
    return driver

def capture_pcap(output_pcap):
    """Captures network traffic using tshark and saves it as a PCAP file."""
    print(f"[info] Starting network traffic...{output_pcap}")
    logging.info(f"Starting PCAP capture: {output_pcap}")
    capture = pyshark.LiveCapture(interface="wlo1", output_file=output_pcap)
    capture.sniff(timeout=180)
    capture.close()
    log_and_print("PCAP capture completed.")

def collect_video_metrics(driver, duration):
    """Collects video resolution and buffer occupancy every second."""
    log_and_print("Starting video metrics collection.")
    start_time = time.time()
    metrics = []
    
    while time.time() - start_time < duration:
        try:
            stats = driver.execute_script(JS_SCRIPT)
            timestamp = time.time()
            metrics.append([timestamp, stats["resolution"], stats["buffer"]])
            logging.info(f"Collected metrics: {stats}")
        except Exception as e:
            logging.warning(f"Failed to collect metrics: {e}")
            metrics.append([time.time(), "N/A", "N/A"])
        
        time.sleep(1)

    log_and_print("Video metrics collection completed.")
    return metrics

def main(url, output_pref, shaping):
    """Main function to run the automation."""
    timestamp = int(time.time())  # Get current timestamp

     # Update output_pref with timestamp
    output_pref_timestamp = f"{output_pref}_{timestamp}"
    
    # Create Analytics directory if it doesn't exist
    analytics_dir = "Analytics"
    create_directory(analytics_dir)

    print(f"[info] Running automation for URL: {url}")
    logging.info(f"Starting automation for URL: {url}")
    output_pcap = os.path.join(analytics_dir, f"{output_pref_timestamp}.pcap")
    output_har = os.path.join(analytics_dir, f"{output_pref_timestamp}.har")
    output_csv = os.path.join(analytics_dir, f"{output_pref_timestamp}.csv")
    output_log = os.path.join(analytics_dir, f"{output_pref_timestamp}.log")
    
    # Remove existing files
    # logging.info("Removing existing files...")
    # clear_files = [output_pcap, output_har, output_csv, output_log]
    # for file in clear_files:
    #     if os.path.exists(file):
    #         os.remove(file)
    # log_and_print("Existing files removed.")


    log_and_print("Initializing HAR capture...")
    proxy.new_har("youtube_video", options={"captureHeaders": True, "captureContent": True})

    driver = setup_webdriver()

    try:
        log_and_print("Navigating to the YouTube video page...")
        driver.get(url)
        time.sleep(3)
        
        # Wait for the video element to appear
        logging.info("Waiting for video element to load...")
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "video")))
        
        video_element = driver.find_element("tag name", "video")
        log_and_print("Video element found, attempting to play video...")
        driver.execute_script("arguments[0].play();", video_element)            # Play the video

        # Capture startup latency
        startup_time = time.time()
        while video_element.get_attribute("paused") == "true":
            time.sleep(0.1)
        startup_time = round(time.time() - startup_time, 2)
        
        print(f"[info] Video started playing. Startup latency: {startup_time} seconds.")
        logging.info(f"Video started playing. Startup latency: {startup_time} seconds.")

        # Start capturing PCAP in a separate thread
        pcap_thread = threading.Thread(target=capture_pcap, args=(output_pcap,))
        pcap_thread.start()

        # Collect video metrics
        metrics = collect_video_metrics(driver, 180)

        # Stop PCAP capture
        pcap_thread.join()

        # Save HAR logs
        log_and_print("Saving HAR logs.")
        har_data = proxy.har
        with open(output_har, "w") as har_file:
            json.dump(har_data, har_file, indent=4)
        logging.info("HAR logs saved.")

        # Save CSV metrics
        log_and_print("Saving video metrics to CSV.")
        df = pd.DataFrame(metrics, columns=["timestamp", "resolution", "buffer_occupancy"])
        df.to_csv(output_csv, index=False)
        logging.info("CSV file saved.")

        # Calculate additional metrics
        avg_resolution = df["resolution"].mode()[0] if not df["resolution"].empty else "N/A"
        rebuffering_ratio = df["buffer_occupancy"].apply(lambda x: 1 if x == 0 else 0).mean()

        # Log output
        log_and_print("Writing log file...")
        with open(output_log, "w") as log_file:
            log_file.write(f"{url},{startup_time},{rebuffering_ratio},{avg_resolution},-1,-1\n" if not shaping 
                        else f"{url},{startup_time},{rebuffering_ratio},{avg_resolution},N/A,N/A\n")
        logging.info("Log file written.")

    finally:
        log_and_print("Closing WebDriver and Proxy...")
        driver.quit()
        proxy.close()
        server.stop()
        log_and_print("Automation process completed.")


# python main.py --url "https://www.youtube.com/watch?v=WevXTsed_oc" --output_pref "486497" --shaping False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--output_pref", required=True, help="Output file prefix")
    parser.add_argument("--shaping", type=bool, default=False, required=True, help="Enable network shaping (True/False)")
    
    args = parser.parse_args()
    main(args.url, args.output_pref, args.shaping)