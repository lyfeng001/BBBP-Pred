#!/usr/bin/env python3
"""
简单测试：比较first和random构象策略
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_load import load_all_bbbp_data
from unimol_tools import MolTrain, MolPredict
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
import traceback


def test_strategy(strategy_name, strategy, random_seed=42):
    """测试单个构象策略"""
    print(f"\n{'='*50}")
    print(f"测试策略: {strategy_name}")
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
            random_seed=random_seed
        )
        
        print(f"训练集大小: {len(train_data['target'])}")
        print(f"验证集大小: {len(valid_data['target'])}")
        print(f"测试集大小: {len(test_data['target'])}")
        
        # 检查SMILES数据
        if 'smiles' in train_data:
            print("✓ 发现SMILES数据")
        else:
            print("✗ 没有发现SMILES数据")
        
        # 初始化模型
        trainer = MolTrain(
            task='classification',
            data_type='molecule',
            epochs=30,  # 减少epoch数以便快速测试
            learning_rate=1e-4,
            batch_size=16,
            early_stopping=10,
            metrics='auc',
            kfold=1  # 不进行交叉验证
        )
        
        # 训练模型
        print("开始训练模型...")
        
        # 格式化训练数据
        train_data_formatted = {
            'atoms': train_data['atoms'],
            'coordinates': train_data['coordinates'],
            'target': [target[0] if isinstance(target, (list, tuple)) else target for target in train_data['target']]
        }
        
        # 添加SMILES数据（使用大写SMILES作为键名）
        if 'smiles' in train_data:
            train_data_formatted['SMILES'] = train_data['smiles']
        
        trainer.fit(data=train_data_formatted)
        
        # 评估
        print("在验证集上评估...")
        
        # 格式化验证数据
        valid_data_formatted = {
            'atoms': valid_data['atoms'],
            'coordinates': valid_data['coordinates']
        }
        
        # 使用MolPredict进行预测
        print(f"加载模型目录: ./exp")
        predictor = MolPredict(load_model='./exp')
        print("模型加载成功，开始预测...")
        valid_pred = predictor.predict(valid_data_formatted)
        
        # 处理预测结果
        valid_pred_array = np.array(valid_pred)
        if valid_pred_array.ndim > 1 and valid_pred_array.shape[1] > 1:
            valid_probabilities = valid_pred_array[:, 1]
            valid_predictions = np.argmax(valid_pred_array, axis=1)
        else:
            valid_pred_array = valid_pred_array.flatten()
            if np.max(valid_pred_array) <= 1.0 and np.min(valid_pred_array) >= 0.0:
                valid_probabilities = valid_pred_array
            else:
                valid_probabilities = 1 / (1 + np.exp(-valid_pred_array))
            valid_predictions = (valid_probabilities > 0.5).astype(int)
        
        valid_labels = [target[0] if isinstance(target, (list, tuple)) else target for target in valid_data['target']]
        valid_accuracy = accuracy_score(valid_labels, valid_predictions)
        valid_auc = roc_auc_score(valid_labels, valid_probabilities)
        
        print("在测试集上评估...")
        
        # 格式化测试数据
        test_data_formatted = {
            'atoms': test_data['atoms'],
            'coordinates': test_data['coordinates']
        }
        
        test_pred = predictor.predict(test_data_formatted)
        
        # 处理预测结果
        test_pred_array = np.array(test_pred)
        if test_pred_array.ndim > 1 and test_pred_array.shape[1] > 1:
            test_probabilities = test_pred_array[:, 1]
            test_predictions = np.argmax(test_pred_array, axis=1)
        else:
            test_pred_array = test_pred_array.flatten()
            if np.max(test_pred_array) <= 1.0 and np.min(test_pred_array) >= 0.0:
                test_probabilities = test_pred_array
            else:
                test_probabilities = 1 / (1 + np.exp(-test_pred_array))
            test_predictions = (test_probabilities > 0.5).astype(int)
        
        test_labels = [target[0] if isinstance(target, (list, tuple)) else target for target in test_data['target']]
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_auc = roc_auc_score(test_labels, test_probabilities)
        
        print(f"\n{strategy_name}策略结果:")
        print(f"验证集 - 准确率: {valid_accuracy:.4f}, ROC-AUC: {valid_auc:.4f}")
        print(f"测试集 - 准确率: {test_accuracy:.4f}, ROC-AUC: {test_auc:.4f}")
        
    except Exception as e:
        print(f"{strategy_name}策略失败: {type(e).__name__}: {str(e)}")
        print(f"详细错误信息：")
        traceback.print_exc()
        
    print("="*50)
    
    return {
        'strategy': strategy_name,
        'valid_accuracy': valid_accuracy,
        'valid_auc': valid_auc,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc
    }


def main():
    """主函数"""
    print("开始比较构象选择策略...")
    
    results = []
    
    # 测试第一个构象策略
    try:
        result1 = test_strategy("first", "first", random_seed=42)
        results.append(result1)
    except Exception as e:
        print(f"First策略失败: {str(e)}")
    
    # 测试随机构象策略
    try:
        result2 = test_strategy("random", "random", random_seed=42)
        results.append(result2)
    except Exception as e:
        print(f"Random策略失败: {str(e)}")
    
    # 比较结果
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("结果比较:")
        print(f"{'='*60}")
        
        for result in results:
            print(f"{result['strategy']}: 验证AUC={result['valid_auc']:.4f}, 测试AUC={result['test_auc']:.4f}")
        
        # 计算改善
        if results[1]['test_auc'] > results[0]['test_auc']:
            improvement = (results[1]['test_auc'] - results[0]['test_auc']) / results[0]['test_auc'] * 100
            print(f"\nRandom策略比First策略在测试集上提升了 {improvement:.2f}%")
        else:
            degradation = (results[0]['test_auc'] - results[1]['test_auc']) / results[0]['test_auc'] * 100
            print(f"\nFirst策略比Random策略在测试集上好 {degradation:.2f}%")


if __name__ == "__main__":
    main() 