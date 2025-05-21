import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm



def transfer_file(tensor_file):
    """
    Function to transfer file from server1 to server2 using scp
    :param file_path: Path of the file to transfer
    """
    # Define your servers and paths
    # tensor_save_path = tensor_file.replace('data_tensor', 'data_tensor1')
    # tensor_dir_name = os.path.dirname(tensor_save_path)

    # video_save_path = tensor_file.replace('data_tensor', 'data').replace('_results.pt', '.mp4')
    # video_dir_name = os.path.dirname(video_save_path)

    # json_save_path = tensor_file.replace('data_tensor', 'data_cropped').replace('_results.pt', '_seg.json')
    # json_dir_name = os.path.dirname(json_save_path)

    # os.makedirs(video_dir_name, exist_ok=True)
    # os.makedirs(json_dir_name, exist_ok=True)
    # os.makedirs(tensor_dir_name, exist_ok=True)


    # Safely quote the paths
    # # remote_user = "ubuntu@89.169.98.73"
    # remote_base_path = "/home/ubuntu/workspace/"
    # # local_user = "root@213.173.108.15"
    # dest_base_path = "/workspace/Dessie/"


    # local_user = "root@213.173.108.15"
    

    video_src = shlex.quote("/home/ubuntu/workspace/" + tensor_file)
    video_dst = shlex.quote('s3://horse-project/' + tensor_file)

    # scp_command = f"scp -i sanchit_lambda.pem {video_src} -i sanchit_lambda.pem -P 14874 {video_dst}"

    # print(video_src)
    # print('='*60)
    # print(video_dst)

    video_command = f'''AWS_ACCESS_KEY_ID=7WC9QMLNPSPBYHI75L59 AWS_SECRET_ACCESS_KEY=tEpSI9qdKDbEzysFh1jLTt58K2eXLy8SmmJU1Pzo aws s3 cp '{video_src}' '{video_dst}' --endpoint-url=https://s3.wasabisys.com'''

    os.system(video_command)


    AWS_ACCESS_KEY_ID=7WC9QMLNPSPBYHI75L59 AWS_SECRET_ACCESS_KEY=tEpSI9qdKDbEzysFh1jLTt58K2eXLy8SmmJU1Pzo aws s3 cp data_tensor s3://horse-project/data_tensor/ --recursive --endpoint-url=https://s3.wasabisys.com


    # try:
    #     subprocess.run(scp_command, shell=True, check=True)
    #     print(f"Successfully transferred: {video_save_path}")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error transferring {video_save_path}: {e}")

    # scp_command = f'scp -i sanchit_lambda.pem "ubuntu@89.169.98.73:/home/ubuntu/workspace/{tensor_file}" -i sanchit_lambda.pem -P 14874 "root@213.173.108.15:/workspace/Dessie/{tensor_save_path}"'

    # # try:
    # #     subprocess.run(scp_command, shell=True, check=True)
    # #     print(f"Successfully transferred: {tensor_file}")
    # # except subprocess.CalledProcessError as e:
    # #     print(f"Error transferring {tensor_file}: {e}")

    # scp_command = f'scp -i sanchit_lambda.pem "ubuntu@89.169.98.73:/home/ubuntu/workspace/{json_save_path}" -i sanchit_lambda.pem -P 14874 "root@213.173.108.15:/workspace/Dessie/{json_save_path}"'

    # # try:
    # #     subprocess.run(scp_command, shell=True, check=True)
    # #     print(f"Successfully transferred: {json_save_path}")
    # # except subprocess.CalledProcessError as e:
    # #     print(f"Error transferring {json_save_path}: {e}")

def main():
    # Read the file list from the text file
    with open("file_path.txt", "r") as f:
        file_paths = f.readlines()

    # Remove any trailing whitespace from the file paths
    file_paths = [file_path.strip() for file_path in file_paths]

    for file_path in tqdm(file_paths):
        # Transfer files sequentially
        transfer_file(file_path)


        exit(0)

    

    # # Use ThreadPoolExecutor to transfer files in parallel
    # with ThreadPoolExecutor(max_workers=20) as executor:
    #     executor.map(transfer_file, file_paths)

if __name__ == "__main__":
    main()
