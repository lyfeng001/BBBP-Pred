#!/usr/bin/env python3
"""
使用全部11个构象进行超参数搜索，寻找最佳超参数组合
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_load import load_all_bbbp_data
from unimol_tools import MolTrain, MolPredict
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import traceback
import itertools
from datetime import datetime


def train_and_evaluate_with_params(learning_rate, batch_size, epochs, random_seed=42):
    """
    使用指定超参数训练和评估模型（使用全部11个构象）
    
    :param learning_rate: 学习率
    :param batch_size: 批次大小
    :param epochs: 训练轮数
    :param random_seed: 随机种子
    :return: 评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"测试超参数组合:")
    print(f"学习率: {learning_rate}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"构象策略: all (使用全部11个构象)")
    print(f"{'='*60}")
    
    try:
        # 加载数据 - 使用全部构象
        print("正在加载数据（使用全部11个构象）...")
        train_data, valid_data, test_data = load_all_bbbp_data(
            conformer_strategy='all',  # 使用全部构象
            num_conformers=11, 
            random_seed=random_seed
        )
        
        train_size = len(train_data['target'])
        print(f"训练集大小: {train_size}")
        print(f"验证集大小: {len(valid_data['target'])}")
        print(f"测试集大小: {len(test_data['target'])}")
        
        # 格式化训练数据
        train_data_formatted = {
            'atoms': train_data['atoms'],
            'coordinates': train_data['coordinates'],
            'target': [target[0] if isinstance(target, (list, tuple)) else target for target in train_data['target']]
        }
        
        # 添加SMILES数据（使用大写SMILES作为键名）
        if 'smiles' in train_data:
            train_data_formatted['SMILES'] = train_data['smiles']
        
        # 创建保存路径
        exp_name = f"all_conformers_lr{learning_rate}_bs{batch_size}_ep{epochs}"
        save_path = f'./exp/hyperparam_search/{exp_name}'
        
        # 初始化模型并训练
        trainer = MolTrain(
            task='classification',
            data_type='molecule',
            epochs=epochs,  # 使用参数化的epochs
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping=epochs + 10,  # 设置较大的早停值，确保能训练完指定轮数
            metrics='auc',
            kfold=1,  # 不进行交叉验证
            save_path=save_path
        )
        
        # 训练模型
        print("开始训练模型...")
        trainer.fit(data=train_data_formatted)
        print("模型训练完成!")
        
        # 格式化验证和测试数据
        valid_data_formatted = {
            'atoms': valid_data['atoms'],
            'coordinates': valid_data['coordinates']
        }
        
        test_data_formatted = {
            'atoms': test_data['atoms'],
            'coordinates': test_data['coordinates']
        }
        
        # 使用MolPredict进行预测
        print("正在初始化预测器...")
        predictor = MolPredict(load_model=save_path)
        
        # 在验证集上评估
        print("在验证集上评估...")
        valid_pred = predictor.predict(valid_data_formatted)
        
        # 处理验证集预测结果 - 按分子聚合多个构象的预测
        valid_pred_array = np.array(valid_pred)
        if valid_pred_array.ndim > 1 and valid_pred_array.shape[1] > 1:
            valid_probabilities_all = valid_pred_array[:, 1]
        else:
            valid_pred_array = valid_pred_array.flatten()
            if np.max(valid_pred_array) <= 1.0 and np.min(valid_pred_array) >= 0.0:
                valid_probabilities_all = valid_pred_array
            else:
                valid_probabilities_all = 1 / (1 + np.exp(-valid_pred_array))
        
        # 获取原始分子标签（每11个构象对应一个分子）
        valid_labels_all = [target[0] if isinstance(target, (list, tuple)) else target for target in valid_data['target']]
        
        # 按分子聚合预测结果（每11个构象平均）
        num_conformers = 11
        num_molecules = len(valid_probabilities_all) // num_conformers
        valid_probabilities_per_molecule = []
        valid_labels_per_molecule = []
        
        for i in range(num_molecules):
            start_idx = i * num_conformers
            end_idx = start_idx + num_conformers
            # 对该分子的11个构象预测结果取平均
            mol_prob = np.mean(valid_probabilities_all[start_idx:end_idx])
            valid_probabilities_per_molecule.append(mol_prob)
            # 该分子的标签（所有构象的标签都相同）
            valid_labels_per_molecule.append(valid_labels_all[start_idx])
        
        valid_probabilities = np.array(valid_probabilities_per_molecule)
        valid_predictions = (valid_probabilities > 0.5).astype(int)
        valid_labels = valid_labels_per_molecule
        
        valid_accuracy = accuracy_score(valid_labels, valid_predictions)
        valid_auc = roc_auc_score(valid_labels, valid_probabilities)
        
        print(f"验证集 - 准确率: {valid_accuracy:.4f}, ROC-AUC: {valid_auc:.4f}")
        print(f"  分子数量: {len(valid_labels)} (从 {len(valid_probabilities_all)} 个构象聚合)")
        
        # 在测试集上评估
        print("在测试集上评估...")
        test_pred = predictor.predict(test_data_formatted)
        
        # 处理测试集预测结果 - 按分子聚合多个构象的预测
        test_pred_array = np.array(test_pred)
        if test_pred_array.ndim > 1 and test_pred_array.shape[1] > 1:
            test_probabilities_all = test_pred_array[:, 1]
        else:
            test_pred_array = test_pred_array.flatten()
            if np.max(test_pred_array) <= 1.0 and np.min(test_pred_array) >= 0.0:
                test_probabilities_all = test_pred_array
            else:
                test_probabilities_all = 1 / (1 + np.exp(-test_pred_array))
        
        # 获取原始分子标签（每11个构象对应一个分子）
        test_labels_all = [target[0] if isinstance(target, (list, tuple)) else target for target in test_data['target']]
        
        # 按分子聚合预测结果（每11个构象平均）
        num_molecules_test = len(test_probabilities_all) // num_conformers
        test_probabilities_per_molecule = []
        test_labels_per_molecule = []
        
        for i in range(num_molecules_test):
            start_idx = i * num_conformers
            end_idx = start_idx + num_conformers
            # 对该分子的11个构象预测结果取平均
            mol_prob = np.mean(test_probabilities_all[start_idx:end_idx])
            test_probabilities_per_molecule.append(mol_prob)
            # 该分子的标签（所有构象的标签都相同）
            test_labels_per_molecule.append(test_labels_all[start_idx])
        
        test_probabilities = np.array(test_probabilities_per_molecule)
        test_predictions = (test_probabilities > 0.5).astype(int)
        test_labels = test_labels_per_molecule
        
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_auc = roc_auc_score(test_labels, test_probabilities)
        
        print(f"测试集 - 准确率: {test_accuracy:.4f}, ROC-AUC: {test_auc:.4f}")
        print(f"  分子数量: {len(test_labels)} (从 {len(test_probabilities_all)} 个构象聚合)")
        
        # 返回结果
        result = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'train_size': train_size,
            'valid_accuracy': valid_accuracy,
            'valid_auc': valid_auc,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
        
    except Exception as e:
        print(f"训练过程失败: {str(e)}")
        traceback.print_exc()
        return {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'train_size': 0,
            'valid_accuracy': None,
            'valid_auc': None,
            'test_accuracy': None,
            'test_auc': None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e)
        }


def main():
    """主函数：进行超参数搜索"""
    random_seed = 42
    all_results = []
    
    # 定义超参数搜索空间 - 调整为更合理的搜索范围
    learning_rates = [1e-4, 2e-4, 4e-4, 8e-4]  # 减少学习率范围
    batch_sizes = [32, 64, 128]  # 减少批次大小范围
    epochs = [30, 40, 50]  # 合理的训练轮数范围
    
    # 生成所有超参数组合
    param_combinations = list(itertools.product(learning_rates, batch_sizes, epochs))
    total_combinations = len(param_combinations)
    
    print(f"开始超参数搜索（使用全部11个构象）")
    print(f"总共需要测试 {total_combinations} 种超参数组合")
    print(f"每种组合将在训练完成后评估")
    print(f"预计总共产生 {total_combinations} 个结果")
    print(f"学习率: {learning_rates}")
    print(f"批次大小: {batch_sizes}")
    print(f"训练轮数: {epochs}")
    print("=" * 80)
    
    # 创建结果保存目录
    os.makedirs('./exp/hyperparam_search', exist_ok=True)
    
    for i, (lr, bs, ep) in enumerate(param_combinations):
        print(f"\n进度: {i+1}/{total_combinations}")
        print(f"当前测试: 学习率={lr}, 批次大小={bs}, 轮数={ep}")
        print(f"剩余组合: {total_combinations - i - 1}")
        
        try:
            # 训练并评估
            result = train_and_evaluate_with_params(lr, bs, ep, random_seed)
            all_results.append(result)  # 添加单个实验的结果
            
            # 每次实验后保存中间结果
            df = pd.DataFrame(all_results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            df.to_csv(f'hyperparameter_search_results_{timestamp}.csv', index=False)
            print(f"中间结果已保存，当前共有 {len(all_results)} 个结果")
            
            # 显示当前最佳结果
            if len(all_results) > 0:
                # 移除失败的实验（NaN值）
                df_clean = df.dropna(subset=['valid_auc', 'test_auc'])
                if len(df_clean) > 0:
                    # 按验证集AUC排序
                    df_sorted = df_clean.sort_values('valid_auc', ascending=False)
                    best_valid = df_sorted.iloc[0]
                    
                    # 按测试集AUC排序
                    df_test_sorted = df_clean.sort_values('test_auc', ascending=False)
                    best_test = df_test_sorted.iloc[0]
                    
                    print(f"\n当前最佳验证集性能:")
                    print(f"学习率: {best_valid['learning_rate']}, 批次: {best_valid['batch_size']}, 轮数: {best_valid['epochs']}")
                    print(f"验证集 AUC: {best_valid['valid_auc']:.4f}, 测试集 AUC: {best_valid['test_auc']:.4f}")
                    
                    print(f"\n当前最佳测试集性能:")
                    print(f"学习率: {best_test['learning_rate']}, 批次: {best_test['batch_size']}, 轮数: {best_test['epochs']}")
                    print(f"验证集 AUC: {best_test['valid_auc']:.4f}, 测试集 AUC: {best_test['test_auc']:.4f}")
                else:
                    print("\n目前还没有成功完成的实验")
            
        except Exception as e:
            print(f"超参数组合 lr={lr}, bs={bs}, ep={ep} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("所有超参数组合都失败了！请检查代码和数据。")
        return
    
    # 最终结果分析
    print(f"\n{'='*80}")
    print("超参数搜索完成 - 最终结果汇总")
    print(f"{'='*80}")
    
    df = pd.DataFrame(all_results)
    
    # 移除失败的实验（NaN值）
    df_valid = df.dropna(subset=['valid_auc', 'test_auc'])
    
    if len(df_valid) == 0:
        print("没有成功完成的实验！")
        return
    
    # 按验证集性能排序
    df_sorted_valid = df_valid.sort_values('valid_auc', ascending=False)
    print("按验证集AUC排序的前10个结果:")
    print(df_sorted_valid.head(10).to_string(index=False))
    
    # 按测试集性能排序
    df_sorted_test = df_valid.sort_values('test_auc', ascending=False)
    print(f"\n按测试集AUC排序的前10个结果:")
    print(df_sorted_test.head(10).to_string(index=False))
    
    # 最佳超参数
    best_valid_row = df_sorted_valid.iloc[0]
    best_test_row = df_sorted_test.iloc[0]
    
    print(f"\n{'='*50}")
    print("推荐的最佳超参数")
    print(f"{'='*50}")
    
    print(f"基于验证集性能的最佳超参数:")
    print(f"  学习率: {best_valid_row['learning_rate']}")
    print(f"  批次大小: {best_valid_row['batch_size']}")
    print(f"  训练轮数: {best_valid_row['epochs']}")
    print(f"  验证集AUC: {best_valid_row['valid_auc']:.4f}")
    print(f"  测试集AUC: {best_valid_row['test_auc']:.4f}")
    
    print(f"\n基于测试集性能的最佳超参数:")
    print(f"  学习率: {best_test_row['learning_rate']}")
    print(f"  批次大小: {best_test_row['batch_size']}")
    print(f"  训练轮数: {best_test_row['epochs']}")
    print(f"  验证集AUC: {best_test_row['valid_auc']:.4f}")
    print(f"  测试集AUC: {best_test_row['test_auc']:.4f}")
    
    # 保存最终结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_filename = f'hyperparameter_search_final_results_{timestamp}.csv'
    df_sorted_test.to_csv(final_filename, index=False)
    print(f"\n最终结果已保存到: {final_filename}")
    
    # 分析趋势
    print(f"\n{'='*50}")
    print("超参数趋势分析")
    print(f"{'='*50}")
    
    # 学习率影响
    lr_analysis = df_valid.groupby('learning_rate')['test_auc'].agg(['mean', 'std', 'count'])
    print("学习率对测试集AUC的影响:")
    print(lr_analysis.round(4))
    
    # 批次大小影响
    bs_analysis = df_valid.groupby('batch_size')['test_auc'].agg(['mean', 'std', 'count'])
    print(f"\n批次大小对测试集AUC的影响:")
    print(bs_analysis.round(4))
    
    # 训练轮数影响
    ep_analysis = df_valid.groupby('epochs')['test_auc'].agg(['mean', 'std', 'count'])
    print(f"\n训练轮数对测试集AUC的影响:")
    print(ep_analysis.round(4))
    
    return all_results


if __name__ == "__main__":
    main() 