import lmdb
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans


def bbbp_load(path, conformer_strategy='first', num_conformers=1, random_seed=42):
    """
    加载BBBP数据集的LMDB文件
    返回包含atoms、coordinates、smiles等字段的字典格式数据
    
    :param path: LMDB文件路径
    :param conformer_strategy: 构象选择策略
        - 'first': 只使用第一个构象（默认）
        - 'random': 随机选择一个构象
        - 'all': 使用所有构象（会扩展数据集）
        - 'cluster': 使用聚类选择代表性构象
        - 'best_diverse': 选择多样性最好的构象
    :param num_conformers: 当strategy为'cluster'或'best_diverse'时，选择的构象数量
    :param random_seed: 随机种子
    """
    np.random.seed(random_seed)
    
    env = lmdb.open(
        path, # your dataset path
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )

    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    
    atoms_list = []
    coordinates_list = []
    labels_list = []
    smiles_list = []
    scaffold_list = []
    
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        
        # 获取原子列表
        atoms = data['atoms']
        
        # 获取所有构象的坐标
        coords = np.array(data['coordinates'])  # (num_conformers, num_atoms, 3)
        
        if conformer_strategy == 'first':
            # 只使用第一个构象
            selected_coords = [coords[0]]
            selected_atoms = [atoms]
        elif conformer_strategy == 'random':
            # 随机选择一个构象
            random_idx = np.random.randint(0, coords.shape[0])
            selected_coords = [coords[random_idx]]
            selected_atoms = [atoms]
        elif conformer_strategy == 'all':
            # 使用所有构象（会扩展数据集大小）
            selected_coords = [coords[i] for i in range(coords.shape[0])]
            selected_atoms = [atoms] * coords.shape[0]
        elif conformer_strategy == 'cluster':
            # 使用聚类选择代表性构象
            selected_coords, selected_atoms = _select_diverse_conformers(
                coords, atoms, num_conformers, method='cluster'
            )
        elif conformer_strategy == 'best_diverse':
            # 选择最多样化的构象
            selected_coords, selected_atoms = _select_diverse_conformers(
                coords, atoms, num_conformers, method='diverse'
            )
        else:
            raise ValueError(f"Unknown conformer strategy: {conformer_strategy}")
        
        # 添加所有选中的构象
        for coord, atom in zip(selected_coords, selected_atoms):
            atoms_list.append(atom)
            coordinates_list.append(coord.tolist())
            
            # 对每个构象都重复相同的标签和SMILES信息
            if 'smi' in data:
                smiles_list.append(data['smi'])
            
            if 'scaffold' in data:
                scaffold_list.append(data['scaffold'])
            
            if 'target' in data:
                labels_list.append(data['target'])
            elif 'label' in data:
                labels_list.append(data['label'])
    
    env.close()
    
    result = {
        'atoms': atoms_list,
        'coordinates': coordinates_list
    }
    
    if smiles_list:
        result['smiles'] = smiles_list
        
    if scaffold_list:
        result['scaffold'] = scaffold_list
    
    if labels_list:
        result['target'] = labels_list
    
    return result


def _select_diverse_conformers(coords, atoms, num_conformers, method='cluster'):
    """
    从多个构象中选择多样化的代表性构象
    
    :param coords: 构象坐标数组 (num_conformers, num_atoms, 3)
    :param atoms: 原子列表
    :param num_conformers: 要选择的构象数量
    :param method: 选择方法 ('cluster' 或 'diverse')
    :return: 选中的构象坐标列表和对应的原子列表
    """
    if coords.shape[0] <= num_conformers:
        # 如果构象数量不超过需要的数量，返回所有构象
        return [coords[i] for i in range(coords.shape[0])], [atoms] * coords.shape[0]
    
    if method == 'cluster':
        # 使用K-means聚类选择代表性构象
        # 将坐标展平用于聚类
        coords_flat = coords.reshape(coords.shape[0], -1)
        
        kmeans = KMeans(n_clusters=num_conformers, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_flat)
        
        # 为每个聚类选择最接近聚类中心的构象
        selected_indices = []
        for i in range(num_conformers):
            cluster_mask = cluster_labels == i
            if not cluster_mask.any():
                continue
            cluster_coords = coords_flat[cluster_mask]
            cluster_center = kmeans.cluster_centers_[i]
            
            # 找到最接近聚类中心的构象
            distances = np.linalg.norm(cluster_coords - cluster_center, axis=1)
            best_idx_in_cluster = np.argmin(distances)
            original_idx = np.where(cluster_mask)[0][best_idx_in_cluster]
            selected_indices.append(original_idx)
        
    elif method == 'diverse':
        # 使用贪心算法选择多样化构象
        selected_indices = [0]  # 从第一个构象开始
        coords_flat = coords.reshape(coords.shape[0], -1)
        
        for _ in range(num_conformers - 1):
            max_min_distance = -1
            best_idx = -1
            
            for i in range(coords.shape[0]):
                if i in selected_indices:
                    continue
                
                # 计算到已选构象的最小距离
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    distance = np.linalg.norm(coords_flat[i] - coords_flat[selected_idx])
                    min_distance = min(min_distance, distance)
                
                # 选择最小距离最大的构象（最远的构象）
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
    
    selected_coords = [coords[i] for i in selected_indices]
    selected_atoms = [atoms] * len(selected_indices)
    
    return selected_coords, selected_atoms


def load_all_bbbp_data(conformer_strategy='first', num_conformers=1, random_seed=42):
    """
    加载所有BBBP数据集（训练集、验证集、测试集）
    
    :param conformer_strategy: 构象选择策略
    :param num_conformers: 构象数量（当strategy需要时）
    :param random_seed: 随机种子
    """
    train_data = bbbp_load("data/bbbp_data/train.lmdb", conformer_strategy, num_conformers, random_seed)
    valid_data = bbbp_load("data/bbbp_data/valid.lmdb", conformer_strategy, num_conformers, random_seed)
    test_data = bbbp_load("data/bbbp_data/test.lmdb", conformer_strategy, num_conformers, random_seed)
    
    return train_data, valid_data, test_data


def inspect_data_sample():
    """
    检查数据样本的格式
    """
    env = lmdb.open(
        "data/bbbp_data/test.lmdb",
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )

    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    
    # 检查多个样本
    for sample_idx, key in enumerate(keys[:3]):
        print(f"\n=== 样本 {sample_idx + 1} ===")
        datapoint_pickled = txn.get(key)
        data = pickle.loads(datapoint_pickled)
        
        atoms = data['atoms']
        coords = np.array(data['coordinates'])
        
        print(f"atoms长度：{len(atoms)}")
        print(f"coordinates形状：{coords.shape}")
        print(f"第一个构象形状：{coords[0].shape}")
        
        # 检查原子列表
        non_empty_atoms = [atom for atom in atoms if atom and atom.strip()]
        print(f"非空原子数量：{len(non_empty_atoms)}")
        
        # 显示所有原子
        print(f"所有原子：{atoms}")
        
        # 检查是否atoms长度和坐标第二维匹配
        print(f"atoms长度 == coords第二维？ {len(atoms) == coords.shape[1]}")
    
    env.close()


if __name__ == "__main__":
    inspect_data_sample()