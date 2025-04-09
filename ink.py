import json

# TODO: do other days in logs

def parse_logs(file_path):
    with open(file_path, 'r') as file:
        file.readline()
        content = file.read()

    logs_section = content.split("Activities log:")[0]
    
    logs = []
    valid = False
    
    for txt in logs_section.splitlines():
        i = txt.strip()
        if len(i) == 0:
            continue
        if i[0] == "{":
            logs.append("{")
            valid = True
        elif i[0] == "}":
            logs[-1] += i
            valid = False
        elif valid:
            logs[-1] += i
        

    parsed_logs = []
    for log in logs:
        try:
            parsed_logs.append(json.loads(log))
        except json.JSONDecodeError as e:
            print(f"Error parsing log: {log}\n{e}")
    
    return parsed_logs

file_path = "backtests/2025-04-08_19-33-15.log"  # Replace with the actual file path
parsed_logs = parse_logs(file_path)

prices = []
diffs = []

# read from prices.txt if it exists
try:
    with open("prices.txt", "r") as f:
        prices = [float(line.strip()) for line in f.readlines()]
except FileNotFoundError:
    for i in parsed_logs:
        i = (eval(i['lambdaLog']))
        ink_dicts = (i[0][3]["SQUID_INK"])

        buy_dict = ink_dicts[0]  # First dictionary in the list
        sell_dict = ink_dicts[1]  # Second dictionary in the list

        # Find the key with the highest value in the buy dictionary
        buy_price = max(buy_dict, key=buy_dict.get)
        buy_quantity = buy_dict[buy_price]

        # Find the key with the highest value in the sell dictionary
        sell_price = max(sell_dict, key=sell_dict.get)
        sell_quantity = sell_dict[sell_price]
        
        prices.append((int(buy_price)+int(sell_price))/2) 

    # save prices to a file
    with open("prices.txt", "w") as f:
        for price in prices:
            f.write(f"{price}\n")

for i in range(len(prices)-2):
    diffs.append((prices[i+2]-prices[i+1], prices[i+1]-prices[i]))

# plot diffs
import matplotlib.pyplot as plt
import numpy as np

import sklearn
jmp = 2900
for n in range(10):
    #diffs = diffs[n*5000:n*5000+5000]

    diffs_array = np.array(diffs[n*jmp:n*jmp+jmp])

    # Randomly subset diffs
    #subset_size = 5000  # Define the size of the subset
    #random_indices = np.random.choice(len(diffs_array), size=subset_size, replace=False)
    #random_subset = diffs_array[random_indices]

    # Extract x and y components from the subset
    diff_x = diffs_array[:, 0]  # First column
    diff_y = diffs_array[:, 1]  # Second column

    correlation_matrix = np.corrcoef(diff_x, diff_y)
    correlation_coefficient = correlation_matrix[0, 1]

    print(f"corr_coefficient: {correlation_coefficient}")
# plot diffs as (x,y)
plt.figure(figsize=(10, 8))
plt.scatter([diff[0] for diff in diffs], [diff[1] for diff in diffs], alpha=0.5)

plt.show()