import os
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime



# ===== Utility Functions =====
def get_IAT(timestamps):
    return [j - i for i, j in zip(timestamps[:-1], timestamps[1:])]


# ===== Create Bidirectional Key =====
def get_bidirectional_key(ip1, port1, ip2, port2, proto):
    # Ensures order-independent key
    return tuple(sorted([(ip1, port1), (ip2, port2)])) + (proto,)

# ===== Step 1: Read and Combine CSVs =====
def read_and_filter_csv(directory='data'):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    print(f"Found {len(files)} CSV files in the directory.\n")

    columns = ["frame.time_epoch", "frame.len", "ip.src", "ip.dst", "ip.proto",
               "udp.srcport", "udp.dstport", "tcp.srcport", "tcp.dstport",
               "tcp.flags", "tcp.flags.syn", "tcp.flags.fin", 
               "dns.qry.name"]

    combined_data = pd.DataFrame()

    for index, file in enumerate(files):
        file_path = os.path.join(directory, file)
        print(f"[{index + 1:03}] Processing File: {os.path.basename(file_path)}")
        data = pd.read_csv(file_path, sep='\t', header=None, names=columns, dtype={4: str, 12: str})

        # Filter rows
        before_drop = len(data)
        data = data.dropna(subset=["frame.len", "ip.src", "ip.dst"])
        data = data[data["ip.proto"].str.match(r"^\d+$", na=False)]
        data["ip.proto"] = data["ip.proto"].astype(int)

        # Keep only IP packets(TCP or UDP)
        data = data[(data["ip.proto"] == 6) | (data["ip.proto"] == 17)]

        # Remove DNS packets
        data = data[data["dns.qry.name"].isna()]
        after_drop = len(data)

        diff = before_drop - after_drop
        print(f"Filtered {diff} rows due to missing frame.len, ip.src or ip.dst OR Protocol is non-IP\n")

        # Label from filename
        data["Label"] = '_'.join(os.path.basename(file_path).split('_')[:2])
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    return combined_data


# ===== Step 2: Flow Extraction =====
def extract_flows(df, udp_timeout=60):
    flows = defaultdict(lambda: {"timestamps": [], "sizes": [], "directions": [], "label": None})
    flow_index = 1
    output = []

    for idx, row in df.iterrows():
        try:
            ts = float(row["frame.time_epoch"])
            size = int(row["frame.len"])
            proto = int(row["ip.proto"])
            src_ip, dst_ip = row["ip.src"], row["ip.dst"]

            if proto == 6:  # TCP
                sport, dport = int(row["tcp.srcport"]), int(row["tcp.dstport"])
                syn_flag = str(row["tcp.flags.syn"]).strip() == '1'
                fin_flag = str(row["tcp.flags.fin"]).strip() == '1'
                label = row["Label"]

                conn_key = get_bidirectional_key(src_ip, sport, dst_ip, dport, "TCP")
                direction = 1 if (src_ip, sport) < (dst_ip, dport) else 0

                flows[conn_key]["timestamps"].append(ts)
                flows[conn_key]["sizes"].append(size)
                flows[conn_key]["directions"].append(direction)
                flows[conn_key]["label"] = label

                if fin_flag:
                    flow_data = flows.pop(conn_key)
                    flow_duration = max(flow_data["timestamps"]) - min(flow_data["timestamps"])
                    output.append({
                        "index": flow_index,
                        "connection": conn_key,
                        "timestamps": flow_data["timestamps"],
                        "sizes": flow_data["sizes"],
                        "directions": flow_data["directions"],
                        "flow_duration": flow_duration,
                        "label_encoded": flow_data["label"]
                    })
                    flow_index += 1

            elif proto == 17:  # UDP
                sport, dport = int(row["udp.srcport"]), int(row["udp.dstport"])
                label = row["Label"]

    
                conn_key = get_bidirectional_key(src_ip, sport, dst_ip, dport, "UDP")
                direction = 1 if (src_ip, sport) < (dst_ip, dport) else 0


                timestamps = flows[conn_key]["timestamps"]
                if timestamps and ts - timestamps[-1] > udp_timeout:
                    flow_data = flows.pop(conn_key)
                    flow_duration = max(flow_data["timestamps"]) - min(flow_data["timestamps"])
                    output.append({
                        "index": flow_index,
                        "connection": conn_key,
                        "timestamps": flow_data["timestamps"],
                        "sizes": flow_data["sizes"],
                        "directions": flow_data["directions"],
                        "flow_duration": flow_duration,
                        "label_encoded": flow_data["label"]
                    })
                    flow_index += 1

                flows[conn_key]["timestamps"].append(ts)
                flows[conn_key]["sizes"].append(size)
                flows[conn_key]["directions"].append(direction)
                flows[conn_key]["label"] = label
        except:
            continue

    # Add remaining UDP flows
    for conn_key, data in flows.items():
        if data["timestamps"]:
            flow_duration = max(data["timestamps"]) - min(data["timestamps"])
            output.append({
                "index": flow_index,
                "connection": conn_key,
                "timestamps": data["timestamps"],
                "sizes": data["sizes"],
                "directions": data["directions"],
                "flow_duration": flow_duration,
                "label_encoded": data["label"]
            })
            flow_index += 1

    return pd.DataFrame(output)


# ===== Step 3: Feature Computation =====
def compute_flow_features(df):
    flow_features = []

    for _, row in df.iterrows():
        ts_list = row['timestamps']
        flow_duration = row['flow_duration']
        size_list = row['sizes']
        dir_list = row['directions']

        fwd_sizes = [s for s, d in zip(size_list, dir_list) if d == 1]
        bwd_sizes = [s for s, d in zip(size_list, dir_list) if d == 0]
        fwd_timestamps = [t for t, d in zip(ts_list, dir_list) if d == 1]
        bwd_timestamps = [t for t, d in zip(ts_list, dir_list) if d == 0]

        flow_iat = get_IAT(ts_list)
        fwd_iat = get_IAT(fwd_timestamps)
        bwd_iat = get_IAT(bwd_timestamps)

        flow_data = {
            'Avg. Packet Size': np.mean(size_list) if size_list else 0,
            'Std Packet Size': np.std(size_list) if len(size_list) > 1 else 0,
            'Max Packet Size': max(size_list) if size_list else 0,
            'Min Packet Size': min(size_list) if size_list else 0,
            'Total Bytes': sum(size_list),
            'Total Packets': len(size_list),
            'Total Fwd Packets': len(fwd_sizes),
            'Total Backward Packets': len(bwd_sizes),
            'Total Length of Fwd Packets': sum(fwd_sizes) if fwd_sizes else 0,
            'Total Length of Bwd Packets': sum(bwd_sizes) if bwd_sizes else 0,
            'Fwd Packet Length Max': max(fwd_sizes) if fwd_sizes else 0,
            'Fwd Packet Length Min': min(fwd_sizes) if fwd_sizes else 0,
            'Fwd Packet Length Mean': np.mean(fwd_sizes) if fwd_sizes else 0,
            'Fwd Packet Length Std': np.std(fwd_sizes) if len(fwd_sizes) > 1 else 0,
            'Bwd Packet Length Max': max(bwd_sizes) if bwd_sizes else 0,
            'Bwd Packet Length Min': min(bwd_sizes) if bwd_sizes else 0,
            'Bwd Packet Length Mean': np.mean(bwd_sizes) if bwd_sizes else 0,
            'Bwd Packet Length Std': np.std(bwd_sizes) if len(bwd_sizes) > 1 else 0,
            'Flow Bytes/s': (sum(size_list) / flow_duration) if flow_duration > 0 else 0,
            'Flow Packets/s': (len(size_list) / flow_duration) if flow_duration > 0 else 0,
            'Flow IAT Mean': np.mean(flow_iat) if flow_iat else 0,
            'Flow IAT Std': np.std(flow_iat) if len(flow_iat) > 1 else 0,
            'Flow IAT Max': max(flow_iat) if flow_iat else 0,
            'Flow IAT Min': min(flow_iat) if flow_iat else 0,
            'Fwd IAT Total': sum(fwd_iat) if fwd_iat else 0,
            'Fwd IAT Mean': np.mean(fwd_iat) if fwd_iat else 0,
            'Fwd IAT Std': np.std(fwd_iat) if len(fwd_iat) > 1 else 0,
            'Fwd IAT Max': max(fwd_iat) if fwd_iat else 0,
            'Fwd IAT Min': min(fwd_iat) if fwd_iat else 0,
            'Bwd IAT Total': sum(bwd_iat) if bwd_iat else 0,
            'Bwd IAT Mean': np.mean(bwd_iat) if bwd_iat else 0,
            'Bwd IAT Std': np.std(bwd_iat) if len(bwd_iat) > 1 else 0,
            'Bwd IAT Max': max(bwd_iat) if bwd_iat else 0,
            'Bwd IAT Min': min(bwd_iat) if bwd_iat else 0,
        }

        flow_features.append(flow_data)

    return pd.DataFrame(flow_features)



# ===== Driver Function =====
def process_all_csv_flows(data_dir='data', udp_timeout=60):
    # Load/Save the combined DataFrame
    output_file = 'combined_data_nw.csv'
    if os.path.exists(output_file):
        df_combined = pd.read_csv(output_file)
        print(f"Found existing combined data file: {output_file}")
    else:
        df_combined = read_and_filter_csv(data_dir)
        df_combined.to_csv(output_file, index=False)
        print(f"Combined data saved to {output_file}")

    # Extract flows and compute features
    flow_df = extract_flows(df_combined, udp_timeout=udp_timeout)
    
    
    feature_df = compute_flow_features(flow_df)

    final_df = pd.concat([flow_df.reset_index(drop=True), feature_df], axis=1)
    return final_df


final_df = process_all_csv_flows(data_dir='data')
final_df.to_csv("flow_features.csv", index=False)