import os
import time
import json
import argparse
import pandas as pd
import pyshark
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from browsermobproxy import Server

# Start BrowserMob Proxy
BROWSER_MOB_PATH = "/path/to/browsermob-proxy/bin/browsermob-proxy"  # Set the correct path
server = Server(BROWSER_MOB_PATH)
server.start()
proxy = server.create_proxy()

# Function to capture network stats using JavaScript
JS_SCRIPT = """
return {
    resolution: document.querySelector('video') ? document.querySelector('video').videoWidth + 'x' + document.querySelector('video').videoHeight : 'N/A',
    buffer: document.querySelector('video') ? document.querySelector('video').buffered.length > 0 ? document.querySelector('video').buffered.end(0) - document.querySelector('video').currentTime : 0 : 0
};
"""

def setup_webdriver():
    """Configures and launches the Selenium WebDriver."""
    chrome_options = Options()
    chrome_options.add_argument("--proxy-server={0}".format(proxy.proxy))
    chrome_options.add_argument("--headless")  # Run in headless mode
    service = Service("/path/to/chromedriver")  # Set the correct path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def capture_pcap(output_pcap):
    """Captures network traffic using tshark and saves it as a PCAP file."""
    capture = pyshark.LiveCapture(interface="eth0", output_file=output_pcap)  # Use "Wi-Fi" on Windows
    capture.sniff(timeout=180)  # 3 minutes
    capture.close()

def collect_video_metrics(driver, duration):
    """Collects video resolution and buffer occupancy every second."""
    start_time = time.time()
    metrics = []

    while time.time() - start_time < duration:
        try:
            stats = driver.execute_script(JS_SCRIPT)
            timestamp = time.time()
            metrics.append([timestamp, stats["resolution"], stats["buffer"]])
        except:
            metrics.append([time.time(), "N/A", "N/A"])
        
        time.sleep(1)

    return metrics



# python main.py --url "https://www.youtube.com/watch?v=WevXTsed_oc" --output_pref "out" --shaping false

def main(url, output_pref, shaping):
    """Main function to run the automation."""
    output_pcap = f"{output_pref}.pcap"
    output_har = f"{output_pref}.har"
    output_csv = f"{output_pref}.csv"
    output_log = f"{output_pref}.log"

    proxy.new_har("youtube_video", options={"captureHeaders": True, "captureContent": True})

    driver = setup_webdriver()
    driver.get(url)

    time.sleep(3)  # Let the page load

    # Play the video
    video_element = driver.find_element("tag name", "video")
    driver.execute_script("arguments[0].play();", video_element)

    # Capture startup latency
    startup_time = time.time()
    while video_element.get_attribute("paused") == "true":
        time.sleep(0.1)
    startup_time = round(time.time() - startup_time, 2)

    # Start capturing PCAP in a separate thread
    import threading
    pcap_thread = threading.Thread(target=capture_pcap, args=(output_pcap,))
    pcap_thread.start()

    # Collect video metrics
    metrics = collect_video_metrics(driver, 180)

    # Stop PCAP capture
    pcap_thread.join()

    # Save HAR logs
    har_data = proxy.har
    with open(output_har, "w") as har_file:
        json.dump(har_data, har_file, indent=4)

    # Save CSV metrics
    df = pd.DataFrame(metrics, columns=["timestamp", "resolution", "buffer_occupancy"])
    df.to_csv(output_csv, index=False)

    # Calculate additional metrics
    avg_resolution = df["resolution"].mode()[0] if not df["resolution"].empty else "N/A"
    rebuffering_ratio = df["buffer_occupancy"].apply(lambda x: 1 if x == 0 else 0).mean()

    # Log output
    with open(output_log, "w") as log_file:
        log_file.write(f"{url},{startup_time},{rebuffering_ratio},{avg_resolution},-1,-1\n" if not shaping else f"{url},{startup_time},{rebuffering_ratio},{avg_resolution},N/A,N/A\n")

    driver.quit()
    proxy.close()
    server.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--output_pref", required=True, help="Output file prefix")
    parser.add_argument("--shaping", action="store_true", help="Enable network shaping")

    args = parser.parse_args()
    main(args.url, args.output_pref, args.shaping)