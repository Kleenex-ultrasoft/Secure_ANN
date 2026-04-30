import re
import os
# import pandas as pd

def parse_log(client_path, server_path):
    metrics = {}
    
    if os.path.exists(client_path):
        with open(client_path, 'r') as f:
            c_content = f.read()
    else:
        c_content = ""
        
    if os.path.exists(server_path):
        with open(server_path, 'r') as f:
            s_content = f.read()
    else:
        s_content = ""
        
    def get_val(pattern, text, default=0.0):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    metrics['Total Time (ms)'] = get_val(r'Total time: ([\d\.]+) ms', c_content)
    metrics['Dist Time (ms)'] = get_val(r'Distance time: ([\d\.]+) ms', c_content)
    metrics['TopK Time (ms)'] = get_val(r'Topk time: ([\d\.]+) ms', c_content)
    metrics['PIR Time (ms)'] = get_val(r'Pir time: ([\d\.]+) ms', c_content)
    
    c_comm = get_val(r'Total comm: ([\d\.]+) MB', c_content)
    s_comm = get_val(r'Total comm: ([\d\.]+) MB', s_content)
    metrics['Total Comm (MB)'] = c_comm + s_comm
    
    return metrics

def parse_dim_log(filepath):
    # N,D,Secret_Index_Time_ms
    # 10000,128,214.04
    if not os.path.exists(filepath):
        return {}
    
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ',' not in line or 'N' in line: continue
            parts = line.strip().split(',')
            if len(parts) == 3:
                n, d, t = parts
                try:
                    data[(int(n), int(d))] = float(t)
                except ValueError:
                    continue
    return data

def estimate_retrieval(dim_data, dataset_N, dataset_D):
    # Measured N=10,000. D varies.
    # Linear scaling in N.
    # Interpolate/Extrapolate D.
    
    # Check if we have D
    measured_N = 10000
    
    # Find closest D
    closest_D = min(dim_data.keys(), key=lambda x: abs(x[1] - dataset_D) if x[0] == measured_N else float('inf'))
    
    if closest_D[0] != measured_N:
        return 0
        
    base_time = dim_data[closest_D]
    base_D = closest_D[1]
    
    # Scale by N
    scale_N = dataset_N / measured_N
    
    # Scale by D if not exact match (linear approx)
    # However, ORAM cost might have constant overhead.
    # Looking at data:
    # 128 -> 214ms
    # 3072 -> 469ms
    # It's sublinear in D? Or constant overhead dominates?
    # 214 + (D-128) * slope?
    # Slope = (469 - 214) / (3072 - 128) = 0.086 ms/dim.
    # Constant = 214 - 128*0.086 = 203ms.
    # Cost(N=10k, D) = 203 + 0.086 * D (LAN)
    # WAN: 3148 -> 3454. Very flat. Bandwidth limited? Or latency?
    # WAN constant overhead is huge.
    
    # Simple linear interpolation for D
    if dataset_D in [k[1] for k in dim_data.keys() if k[0] == measured_N]:
        time_for_10k = dim_data[(measured_N, dataset_D)]
    else:
        # Linear interp/extrap using 128 and 3072
        d1, t1 = 128, dim_data[(measured_N, 128)]
        d2, t2 = 3072, dim_data[(measured_N, 3072)]
        slope = (t2 - t1) / (d2 - d1)
        time_for_10k = t1 + slope * (dataset_D - d1)
        
    est_time = time_for_10k * scale_N
    return est_time

config = [
    ('sift', 'lan', 'OpenPanther/logs/sift_client_lan.log', 'OpenPanther/logs/sift_server_lan.log', 'logs/microbench_dim_lan.log'),
    ('sift', 'wan', 'OpenPanther/logs/sift_client_wan.log', 'OpenPanther/logs/sift_server_wan.log', 'logs/microbench_dim_wan.log'),
    ('deep10m', 'lan', 'OpenPanther/logs/deep10m_client_lan.log', 'OpenPanther/logs/deep10m_server_lan.log', 'logs/microbench_dim_lan.log'),
    ('deep10m', 'wan', 'OpenPanther/logs/deep10m_client_wan.log', 'OpenPanther/logs/deep10m_server_wan.log', 'logs/microbench_dim_wan.log')
]

print("Dataset,Mode,Original_Total_Time(s),Original_Comm(MB),New_Retrieval_Time(s),New_Total_Time(s)")

for ds, mode, c_log, s_log, dim_log in config:
    m = parse_log(c_log, s_log)
    dim_data = parse_dim_log(dim_log)
    
    if ds == 'sift':
        N = 87127
        D = 2560 # Centroid Dim? No, cluster size is variable, max 20*128.
        # Actually in Panther, points are retrieved.
        # Max cluster size 20. Dim 128.
        # Element size = (128 + overhead) * 20.
        # Effectively we retrieve a block of size D = 20 * 128 = 2560 (approx).
    else:
        N = 500000
        D = 3840 # 40 * 96
        
    ret_time_ms = estimate_retrieval(dim_data, N, D)
    
    # New Total = Dist + TopK + Retrieval
    new_total_ms = m['Dist Time (ms)'] + m['TopK Time (ms)'] + ret_time_ms
    
    print(f"{ds},{mode},{m['Total Time (ms)']/1000:.2f},{m['Total Comm (MB)']:.1f},{ret_time_ms/1000:.2f},{new_total_ms/1000:.2f}")

print("\n--- Dimensionality Curse (LAN, N=10000) ---")
print("Dimension,Time(ms)")
lan_data = parse_dim_log('logs/microbench_dim_lan.log')
for k in sorted(lan_data.keys()):
    print(f"{k[1]},{lan_data[k]}")

