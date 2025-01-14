import subprocess
import pandas as pd
import os
import time

tic = time.time()

def download_files_with_aws_s3_cp(date_str):
    bucket_path = f"s3://noaa-hrrr-bdp-pds/hrrr.{date_str}/*"
    local_dir = f'/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp/data/hrrr.{date_str}'
    os.makedirs(local_dir, exist_ok=True)
    includes = [
        f"*conus*t{HH:02}z*wrfsfcf{FF:02}*"
        for HH in range(0, 25, 6)  # Hours from 00 to 23
        for FF in range(0, 25, 3)  # Forecast hours from 00 to 24 with 3-hour increments
    ]
    
    # Construct the base command
    command = [
        '/home/tge/s5cmd/s5cmd', '--no-sign-request', '--numworkers 8', '--stat', 'cp', 
    ]
    
    # Add include patterns to the command
    for include in includes:
        command.extend(['--include', f'"{include}"'])
    command.extend([bucket_path, local_dir,])
    
    # Join the command list into a single string
    command_str = ' '.join(command)
    
    # Execute the command
    process = subprocess.run(command_str, shell=True)
    if process.returncode == 0:
        print(f"Files for {date_str} downloaded successfully. {time.time()-tic} seconds elapsed time. ")
    else:
        print(f"Failed to download files for {date_str}. Return code: {process.returncode}")

if __name__ == "__main__":
    # Generate date range from December 1, 2020 to July 31, 2024
    start_date = "2020-12-02"
    end_date = "2024-07-31"
    dates = pd.date_range(start=start_date, end=end_date, freq='D').strftime("%Y%m%d")
    for date_str in dates:
        download_files_with_aws_s3_cp(date_str)