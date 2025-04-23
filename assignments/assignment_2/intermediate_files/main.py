import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

#--------------CONSTANTS-----------------
DATA_PATH = "dataC"
RESULTS_DIR = "results_final"
GROUPED_JSON_FILE = os.path.join(RESULTS_DIR, "grouped_classes.json")
OUTPUT_FLOWS_FILE = os.path.join(RESULTS_DIR, "flows.csv")
UDP_TIMEOUT = 30            # seconds, can be adjusted



# Function to group files by class name
def group_files_by_class(folder_path):
    print(f"Grouping files in folder: {folder_path}")

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]              # Get list of all CSV files
    if not files:
        print("No CSV files found in the specified folder.")
        return

    grouped_files = defaultdict(list)                                               # Dictionary to hold grouped files

    for file in files:
        # Extract the class name before '_capture'
        class_name = '_'.join(file.split('_')[:2])                                  # vpn_skype-chat or nonvpn_rdp, etc.
        grouped_files[class_name].append(file)

    # Print grouped classes
    # for class_name, file_list in grouped_files.items():
        # print(f"\nClass: {class_name}")
        # for file in file_list:
            # print(f"  - {file}")

    # Save groupings to JSON file
    with open(GROUPED_JSON_FILE, 'w') as f:
        json.dump(grouped_files, f, indent=4)

    print(f"Grouped classes saved to '{GROUPED_JSON_FILE}'")
    return grouped_files

# Function to get a bidirectional key for the connection, {sets lower IP/port first}
# ensures that the connection is treated the same regardless of direction
def get_bidirectional_key(src_ip, src_port, dst_ip, dst_port, protocol):
    if (src_ip, src_port) <= (dst_ip, dst_port):
        return (src_ip, src_port, dst_ip, dst_port, protocol)
    else:
        return (dst_ip, dst_port, src_ip, src_port, protocol)

# Function to process files and extract flows
def process_files(grouped_files, folder_path, output_file):
    flow_index = 1
    total_flows = 0
    all_flows = []
    ctr = 0

    # Iterate through each class and its corresponding files
    for label, files in grouped_files.items():
        # Initialize a dictionary to hold flow data
        # Each flow is defined by 5-tuple (src_ip, src_port, dst_ip, dst_port, protocol)
        flows = defaultdict(lambda: {
            "timestamps": [], "sizes": [], "directions": [],
            "label": label, "last_ts": None, "proto": None,
            "first_seen": None, "syn_seen": False, "initiator": None
        })

        for index, file in enumerate(files):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path, sep='\t', header=None, names=[
                    "frame.time_epoch", "frame.len", "ip.src", "ip.dst", "ip.proto",
                    "udp.srcport", "udp.dstport", "tcp.srcport", "tcp.dstport",
                    "tcp.flags", "tcp.flags.syn", "tcp.flags.fin", 
                    "dns.qry.name"], dtype={4:str, 12: str})

                print(ctr)                  # display running count of files processed
                ctr += 1
                print(f"\n[{index + 1:02}/{len(files):02}] Processing file: {os.path.basename(file_path)}")
                # print("Initial packets:", len(df))

                # Drop rows with missing frame.len, ip.src or ip.dst
                df = df.dropna(subset=["frame.len", "ip.src", "ip.dst"])
                # print(len(df), "packets after dropping NaN values")
                
                # Also, ensure that ip.proto is numeric
                df = df[df["ip.proto"].str.match(r"^\d+$", na=False)]
                df["ip.proto"] = df["ip.proto"].astype(int)        
                # print(len(df), "packets after filtering for numeric ip.proto")

                # Filter for TCP and UDP packets only
                df = df[(df["ip.proto"] == 6) | (df["ip.proto"] == 17)]
                # print(len(df), "packets after filtering for TCP/UDP")

                # Also, filter out DNS queries (where dns.qry.name is not NaN)
                df = df[df["dns.qry.name"].isna()]
                # print(len(df), "packets after filtering out DNS queries")

                # Convert relevant columns to appropriate types
                df["frame.time_epoch"] = df["frame.time_epoch"].astype(float)
                df["frame.len"] = df["frame.len"].astype(int)

                # print(f"Class: {label}, Number of packets: {len(df)}")

                # Iterate through the DataFrame rows to extract flow data
                # For each packet, check if it's TCP or UDP and process accordingly
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{label}-{os.path.basename(file)}"):
                    ts = row["frame.time_epoch"]
                    size = row["frame.len"]
                    proto = row["ip.proto"]
                    src_ip = row["ip.src"]
                    dst_ip = row["ip.dst"]
                    direction = None                # 1 for src->dst, 0 for dst->src

                    if proto == 6:  # TCP
                        sport = int(row["tcp.srcport"]) if pd.notna(row["tcp.srcport"]) else 0
                        dport = int(row["tcp.dstport"]) if pd.notna(row["tcp.dstport"]) else 0
                        syn_flag = str(row["tcp.flags.syn"]).strip() == '1'
                        fin_flag = str(row["tcp.flags.fin"]).strip() == '1'

                        # Get the connection key for bidirectional flow
                        # and determine the direction based on IP/port values
                        conn_key = get_bidirectional_key(src_ip, sport, dst_ip, dport, "TCP")

                        # Initialize flow on first sight
                        flow = flows[conn_key]
                        flow["proto"] = "TCP"
                        if flow["first_seen"] is None:
                            flow["first_seen"] = ts
                            if syn_flag:
                                flow["initiator"] = (src_ip, sport)             # src_ip, sport is the initiator of the connection

                        # Set dir. for the packet
                        if flow["initiator"]:
                            direction = 1 if (src_ip, sport) == flow["initiator"] else 0
                        else:
                            direction = 1 if (src_ip, sport) < (dst_ip, dport) else 0

                        # Append the current packet to the flow
                        flow["timestamps"].append(ts)
                        flow["sizes"].append(size)
                        flow["directions"].append(direction)

                        # Close the flow if FIN flag is set
                        if fin_flag:
                            all_flows.append({
                                "index": flow_index,
                                "connection": conn_key,
                                "timestamps": flow["timestamps"],
                                "sizes": flow["sizes"],
                                "directions": flow["directions"],
                                "label": flow["label"]
                            })
                            flow_index += 1
                            del flows[conn_key]  # Remove the flow from the dictionary

                    elif proto == 17:  # UDP
                        sport = int(row["udp.srcport"]) if pd.notna(row["udp.srcport"]) else 0
                        dport = int(row["udp.dstport"]) if pd.notna(row["udp.dstport"]) else 0

                        conn_key = get_bidirectional_key(src_ip, sport, dst_ip, dport, "UDP")
                        flow = flows[conn_key]

                        if flow["first_seen"] is None:
                            flow["first_seen"] = ts
                            flow["initiator"] = (src_ip, sport)
                        flow["proto"] = "UDP"
                        
                        # Check if the flow has been inactive for more than UDP_TIMEOUT seconds or there exists a previous packet for this connection
                        # If so, close the flow and save it to all_flows
                        last_ts = flow["last_ts"]
                        if last_ts is not None and ts - last_ts > UDP_TIMEOUT:
                            all_flows.append({
                                "index": flow_index,
                                "connection": conn_key,
                                "timestamps": flow["timestamps"],
                                "sizes": flow["sizes"],
                                "directions": flow["directions"],
                                "label": flow["label"]
                            })
                            flow_index += 1

                            # Start a new flow after closing the old one
                            flows[conn_key] = {
                                "timestamps": [],
                                "sizes": [],
                                "directions": [],
                                "label": label,
                                "last_ts": ts,
                                "proto": "UDP",
                                "first_seen": ts,
                                "initiator": (src_ip, sport)
                            }

                        # Append the current packet to the flow
                        direction = 1 if (src_ip, sport) == flow["initiator"] else 0
                        flow["timestamps"].append(ts)
                        flow["sizes"].append(size)
                        flow["directions"].append(direction)
                        
                        # Update the last timestamp for the flow
                        # used to check for inactivity in the next iteration
                        flow["last_ts"] = ts if flow["last_ts"] is None else max(flow["last_ts"], ts)

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        # After all files of a class are processed, flush remaining flows
        for conn_key, flow_data in flows.items():
            all_flows.append({
                "index": flow_index,
                "connection": conn_key,
                "timestamps": flow_data["timestamps"],
                "sizes": flow_data["sizes"],
                "directions": flow_data["directions"],
                "label": flow_data["label"]
            })
            flow_index += 1 

        # Save flows from this file
        total_flows += len(all_flows)
        df_flows = pd.DataFrame(all_flows)
        write_header = not os.path.exists(output_file)
        df_flows.to_csv(output_file, mode='a', header=write_header, index=False)
    
        print(f"#Flows for class '{label}': {len(all_flows)}\n")
        all_flows.clear()  # Clear the list for the next class


    print(f"\n\nProcessed and found {total_flows} flows.")
    print(f"Flow data saved to '{output_file}'")



#--------------Driver code---------------         
# python main.py
if __name__ == '__main__':
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Check if grouped_classes.json already exists, 
    # if so, load it into dictionary else compute it
    if os.path.exists(GROUPED_JSON_FILE):
        with open(GROUPED_JSON_FILE, 'r') as f:
            grouped_files = json.load(f)
        print(f"Loaded existing groupings from '{GROUPED_JSON_FILE}'")
    else:
        # If the file doesn't exist, compute the groupings
        print(f"'{GROUPED_JSON_FILE}' not found. Computing groupings for dir.: {DATA_PATH}")
        grouped_files = group_files_by_class(DATA_PATH)


    # Print all classes(20) in the dataset
    print("Classes:", list(grouped_files.keys()))


    # Now, compute flows from the grouped files or load them from the existing file
    if not os.path.exists(OUTPUT_FLOWS_FILE):
        print(f"\n'{OUTPUT_FLOWS_FILE}' not found. Extracting flows from files.")
        process_files(grouped_files, DATA_PATH, OUTPUT_FLOWS_FILE)
    else:
        print(f"\n'{OUTPUT_FLOWS_FILE}' already exists. Skipping flow extraction.")
        flows = pd.read_csv(OUTPUT_FLOWS_FILE)
        print(f"Loaded {len(flows)} flows from '{OUTPUT_FLOWS_FILE}'")


    print("\nFlow data columns and their datatypes: ")
    print(flows.columns)
    print(flows.dtypes)