import subprocess
import itertools
import csv
import re
import os

edges = [0, 0.3, 0.5, 0.8, 1, 1.3, 1.5, 1.8, 2]
limits = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
script_path = r'.\\Prosperity\\trader_tests\\Trader_gridsearch.py'
output_file = 'kelp_results.csv'
kelp_pattern = re.compile(r'KELP:\s*([\d,]+)')

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['edge', 'limit', 'kelp_-2', 'kelp_-1', 'kelp_0'])

    for edge, limit in itertools.product(edges, limits):
        env = os.environ.copy()
        env["EDGE"] = str(edge)
        env["LIMIT"] = str(limit)
        cmd = [
            'prosperity3bt',
            script_path,
            '1',  # still needed as your framework expects this
            '--print',
            '--no-out',
            '--match-trades',
            'worse'
        ]
        
        print(f"\n>>> Running with edge={edge}, limit={limit}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        #print(result.stdout)
        print(result.stdout)
        kelp_values = kelp_pattern.findall(result.stdout)
        kelp_values = [int(k.replace(',', '')) for k in kelp_values]
        kelp_trim = kelp_values[:3]
        print(f"KELP values: {kelp_trim}")

        writer.writerow([edge, limit] + kelp_trim + [sum(kelp_trim)])
        if result.stderr:
            print("ERROR:")
            print(result.stderr)