import subprocess
import pandas as pd
import os
import time

tic = time.time()

def download_files_with_aws_s3_cp(date_str):
    for hh in [0,6,12,18]:
        bucket_path = f"s3://noaa-gefs-pds/gefs.{date_str}/{hh:02d}/atmos/pgrb2ap5/*"
        local_dir = f'/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp/data/gefs.{date_str}'
        os.makedirs(local_dir, exist_ok=True)
        includes = [
            f"*gec00*0p50.f0{FF:02d}"
            for FF in range(0, 25, 3)  # Forecast hours from 00 to 24 with 3-hour increments
        ]
        
        # Construct the base command
        command = [
            '/home/tge/s5cmd/s5cmd', '--no-sign-request', '--stat', 'cp', 
        ]
        # Add include patterns to the command
        for include in includes:
            command.extend(['--include', f'"{include}"'])
        command.extend([bucket_path, local_dir,])      
        # Join the command list into a single string
        command_str = ' '.join(command)
        # Execute the command
        process = subprocess.run(command_str, shell=True)

        bucket_path = f"s3://noaa-gefs-pds/gefs.{date_str}/{hh:02d}/atmos/pgrb2sp25/*"
        local_dir = f'/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp/data/gefs.{date_str}'
        os.makedirs(local_dir, exist_ok=True)
        includes = [
            f"*gec00*0p25.f0{FF:02d}"
            for FF in range(0, 25, 3)  # Forecast hours from 00 to 24 with 3-hour increments
        ]
        
        # Construct the base command
        command = [
            '/home/tge/s5cmd/s5cmd', '--no-sign-request', '--stat', 'cp', 
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
    end_date = "2024-11-30"
    dates = pd.date_range(start=start_date, end=end_date, freq='D').strftime("%Y%m%d")
    for date_str in dates:
        download_files_with_aws_s3_cp(date_str)