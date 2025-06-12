#!/usr/bin/env python3
"""
比较不同构象选择策略对UniMol模型性能的影响
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_load import load_all_bbbp_data
from unimol_tools import MolTrain, MolPredict
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
import pandas as pd
import traceback


def aggregate_conformer_predictions(predictions, labels, num_conformers):
    """
    将多个构象的预测结果聚合到分子级别
    
    :param predictions: 所有构象的预测结果
    :param labels: 所有构象的标签（重复的）
    :param num_conformers: 每个分子的构象数量
    :return: 分子级别的预测概率和标签
    """
    predictions = np.array(predictions)
    
    # 处理预测结果格式
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        probabilities_all = predictions[:, 1]
    else:
        predictions_flat = predictions.flatten()
        if np.max(predictions_flat) <= 1.0 and np.min(predictions_flat) >= 0.0:
            probabilities_all = predictions_flat
        else:
            probabilities_all = 1 / (1 + np.exp(-predictions_flat))
    
    labels_all = [target[0] if isinstance(target, (list, tuple)) else target for target in labels]
    
    # 如果只有一个构象，直接返回
    if num_conformers == 1:
        return probabilities_all, labels_all
    
    # 按分子聚合预测结果
    num_molecules = len(probabilities_all) // num_conformers
    probabilities_per_molecule = []
    labels_per_molecule = []
    
    for i in range(num_molecules):
        start_idx = i * num_conformers
        end_idx = start_idx + num_conformers
        # 对该分子的多个构象预测结果取平均
        mol_prob = np.mean(probabilities_all[start_idx:end_idx])
        probabilities_per_molecule.append(mol_prob)
        # 该分子的标签（所有构象的标签都相同）
        labels_per_molecule.append(labels_all[start_idx])
    
    return np.array(probabilities_per_molecule), labels_per_molecule


def train_and_evaluate(strategy, num_conformers=1, random_seed=42):
    """
    使用指定的构象策略训练和评估模型
    
    :param strategy: 构象选择策略
    :param num_conformers: 构象数量
    :param random_seed: 随机种子
    :return: 验证集和测试集的评估结果
    """
    print(f"\n{'='*50}")
    print(f"测试构象策略: {strategy}")
    if strategy in ['cluster', 'best_diverse']:
        print(f"构象数量: {num_conformers}")
    elif strategy == 'all':
        print(f"使用所有构象 (11个)")
    print(f"{'='*50}")
    
    valid_accuracy = None
    valid_auc = None
    test_accuracy = None
    test_auc = None
    
    try:
        # 加载数据
        print("正在加载数据...")
        train_data, valid_data, test_data = load_all_bbbp_data(
            conformer_strategy=strategy, 
            num_conformers=num_conformers, 
            random_seed=random_seed
        )
        
        print(f"训练集大小: {len(train_data['target'])}")
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
        
        # 初始化模型
        trainer = MolTrain(
            task='classification',
            data_type='molecule',
            epochs=40,  # 减少epochs以加快测试
            learning_rate=4e-4,
            batch_size=128,
            early_stopping=10,
            metrics='auc',
            kfold=1,  # 不进行交叉验证
            save_path=f'./exp/conformer_test_{strategy}_{num_conformers}'
        )
        
        # 训练模型
        print("开始训练模型...")
        trainer.fit(data=train_data_formatted)
        
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
        predictor = MolPredict(load_model=f'./exp/conformer_test_{strategy}_{num_conformers}')
        
        # 在验证集上评估
        print("在验证集上评估...")
        valid_pred = predictor.predict(valid_data_formatted)
        
        # 使用辅助函数处理多构象预测结果聚合
        actual_num_conformers = 11 if strategy == 'all' else num_conformers
        valid_probabilities, valid_labels = aggregate_conformer_predictions(
            valid_pred, valid_data['target'], actual_num_conformers
        )
        valid_predictions = (valid_probabilities > 0.5).astype(int)
        
        valid_accuracy = accuracy_score(valid_labels, valid_predictions)
        valid_auc = roc_auc_score(valid_labels, valid_probabilities)
        
        print(f"验证集 - 准确率: {valid_accuracy:.4f}, ROC-AUC: {valid_auc:.4f}")
        if actual_num_conformers > 1:
            total_conformers = len(valid_pred)
            print(f"  分子数量: {len(valid_labels)} (从 {total_conformers} 个构象聚合)")
        
        # 在测试集上评估
        print("在测试集上评估...")
        test_pred = predictor.predict(test_data_formatted)
        
        # 使用辅助函数处理多构象预测结果聚合
        test_probabilities, test_labels = aggregate_conformer_predictions(
            test_pred, test_data['target'], actual_num_conformers
        )
        test_predictions = (test_probabilities > 0.5).astype(int)
        
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_auc = roc_auc_score(test_labels, test_probabilities)
        
        print(f"测试集 - 准确率: {test_accuracy:.4f}, ROC-AUC: {test_auc:.4f}")
        if actual_num_conformers > 1:
            total_conformers_test = len(test_pred)
            print(f"  分子数量: {len(test_labels)} (从 {total_conformers_test} 个构象聚合)")
        
    except Exception as e:
        print(f"策略 {strategy} (num_conformers={num_conformers}) 失败: {str(e)}")
        traceback.print_exc()
    
    return {
        'strategy': strategy,
        'num_conformers': num_conformers,
        'train_size': len(train_data['target']) if 'train_data' in locals() else 0,
        'valid_accuracy': valid_accuracy,
        'valid_auc': valid_auc,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc
    }


def main():
    """主函数：比较不同构象策略"""
    random_seed = 42
    results = []
    
    # 测试策略列表 - 从轻量级到重量级
    strategies_to_test = [
        ('first', 1),           # 只使用第一个构象
        ('random', 1),          # 随机选择一个构象 
        ('cluster', 3),         # 聚类选择3个构象 
        ('best_diverse', 3),    # 多样性选择3个构象 
        ('cluster', 5),         # 聚类选择5个构象 
        ('best_diverse', 5),    # 多样性选择5个构象 
    ]

    
    print(f"将测试 {len(strategies_to_test)} 种构象策略...")
    
    for i, (strategy, num_conformers) in enumerate(strategies_to_test):
        print(f"\n进度: {i+1}/{len(strategies_to_test)}")
        try:
            result = train_and_evaluate(strategy, num_conformers, random_seed)
            results.append(result)
            
            # 保存中间结果
            df = pd.DataFrame(results)
            df.to_csv('conformer_strategy_results.csv', index=False)
            print(f"中间结果已保存到 conformer_strategy_results.csv")
            
            # 显示当前最佳结果
            if len(results) > 1:
                best_so_far = max(results, key=lambda x: x['test_auc'])
                print(f"目前最佳策略: {best_so_far['strategy']} (构象数: {best_so_far['num_conformers']})")
                print(f"最佳测试 ROC-AUC: {best_so_far['test_auc']:.4f}")
            
        except Exception as e:
            print(f"策略 {strategy} (num_conformers={num_conformers}) 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("所有策略都失败了！请检查代码和数据。")
        return
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("所有策略的结果汇总:")
    print(f"{'='*80}")
    
    df = pd.DataFrame(results)
    # 按测试AUC排序
    df_sorted = df.sort_values('test_auc', ascending=False)
    print(df_sorted.to_string(index=False))
    
    # 找到最佳策略
    best_valid = df.loc[df['valid_auc'].idxmax()]
    best_test = df.loc[df['test_auc'].idxmax()]
    
    print(f"\n最佳验证集性能:")
    print(f"策略: {best_valid['strategy']} (构象数: {best_valid['num_conformers']})")
    print(f"验证集 ROC-AUC: {best_valid['valid_auc']:.4f}")
    print(f"测试集 ROC-AUC: {best_valid['test_auc']:.4f}")
    
    print(f"\n最佳测试集性能:")
    print(f"策略: {best_test['strategy']} (构象数: {best_test['num_conformers']})")
    print(f"验证集 ROC-AUC: {best_test['valid_auc']:.4f}")
    print(f"测试集 ROC-AUC: {best_test['test_auc']:.4f}")
    
    # 分析性能提升
    first_result = next((r for r in results if r['strategy'] == 'first'), None)
    if first_result:
        print(f"\n与baseline (first) 策略的比较:")
        baseline_auc = first_result['test_auc']
        for result in results:
            if result['strategy'] != 'first':
                improvement = ((result['test_auc'] - baseline_auc) / baseline_auc) * 100
                print(f"{result['strategy']} (构象数: {result['num_conformers']}): "
                      f"测试AUC提升 {improvement:+.2f}%")
    
    # 保存最终结果
    df_sorted.to_csv('conformer_strategy_results_final.csv', index=False)
    print(f"\n最终结果已保存到 conformer_strategy_results_final.csv")
    
    return results


if __name__ == "__main__":
    main() 