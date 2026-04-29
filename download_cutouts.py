import os
import requests
import illustris_python as il
from tqdm import tqdm 

cutout_output_dir = "/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts"
basePath = "/public/home/zju_visitor/LiuYuanhao/tng_data/output"
snapNum = 33

API_KEY = "efc5aaa1f009fd7189ca64c564681762"  
headers = {"api-key": API_KEY}
base_url = "https://www.tng-project.org/api/"
sim_name = "TNG50-1"

def download_stream_environment(snapNum, subhalo_id):
    os.makedirs(cutout_output_dir, exist_ok=True)
    save_path = os.path.join(cutout_output_dir, "cutout_snap{:03d}_sh{}.hdf5".format(snapNum, subhalo_id))
    temp_path = save_path + ".part" 
    
    if os.path.exists(save_path):
        print(f"文件已存在，跳过下载: {save_path}")
        return save_path

    try:
        all_grnrs = il.groupcat.loadSubhalos(basePath, snapNum, fields=['SubhaloGrNr'])
        parent_halo_id = all_grnrs[subhalo_id]
    except Exception as e:
        print(f"读取本地星表失败，请检查 {basePath} 下是否存在 groups_{snapNum:03d} 文件夹！")
        raise e

    gas_fields = "Coordinates,Masses,Density,InternalEnergy,ElectronAbundance,StarFormationRate,Velocities,GFM_Metallicity"
    query_url = f"{base_url}{sim_name}/snapshots/{snapNum}/halos/{parent_halo_id}/cutout.hdf5?gas={gas_fields}"
    
    print(f"正在从云端拉取母晕 (Halo ID: {parent_halo_id}) 的包围盒...")
    
    try:
        with requests.get(query_url, headers=headers, stream=True, timeout=(10, 300)) as r:
            r.raise_for_status() 
            total_size = int(r.headers.get('content-length', 0))
            
            with open(temp_path, 'wb') as f, tqdm(
                desc=f"ShID {subhalo_id:5d}", 
                total=total_size, 
                unit='iB', 
                unit_scale=True, 
                unit_divisor=1024,
                leave=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        
        os.rename(temp_path, save_path)
        print(f"成功，数据保存在: {save_path}\n") 
        
    except Exception as err:
        print(f"下载失败 Subhalo {subhalo_id}: {err}")
        # 如果下载失败，清理可能残留的临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise err

    return save_path

subhalo_ids = [ 59075,  60750,  71129,  72910,  75667,  76827,  80399,  82445,
        83577,  85522,  86525,  90626,  91433,  94857,  95754,  96548,
        97357,  98128,  99303, 100780, 101498, 102284, 102999, 104564,
       106287, 107194, 107752, 108487, 109048, 109728, 110543, 111196,
       113348, 114745, 115247, 117098, 117546, 118044, 118608, 119023,
       119503, 121252, 122298, 123020, 123607, 124044, 124587, 125841,
       126923, 127580]

if __name__ == "__main__":
    print(f"准备为 {len(subhalo_ids)} 个目标星系批量下载局部环境...")

    for shID in subhalo_ids:
        download_stream_environment(snapNum, shID)

    print("全部环境数据下载完毕。")