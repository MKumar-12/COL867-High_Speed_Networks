import json
from haralyzer import HarParser, HarPage

# Load HAR file
File_Path = 'C:\\Users\\manis\\OneDrive\\Desktop\\HSN_sem4\\assignments\\assignment_1\\intermediate_files\\yt_video.har'
with open(File_Path, 'r') as f:
    har_data = json.load(f)

# Parse HAR data
har_parser = HarParser(har_data)

# Analyze the first page
page = har_parser.pages[0]

# Print basic information
print(f"Page ID: {page.page_id}")
print(f"Page Title: {page.title}")
print(f"Page Started DateTime: {page.startedDateTime}")
# print(f"Page Load Time: {page.page_load_time} ms")

# Print details of each request
for entry in page.entries:
    request = entry['request']
    response = entry['response']
    print(f"Request URL: {request['url']}")
    print(f"Request Method: {request['method']}")
    print(f"Response Status: {response['status']}")
    print(f"Response Content Type: {response['content']['mimeType']}")
    print("-" * 40)