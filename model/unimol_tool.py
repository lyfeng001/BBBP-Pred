import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unimol_tools import MolTrain, MolPredict
from data.data_load import load_all_bbbp_data
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


def train_and_test_bbbp():
    """
    使用BBBP数据集训练和测试UniMol模型
    """
    print("正在加载BBBP数据集...")
    
    # 加载训练集、验证集和测试集
    train_data, valid_data, test_data = load_all_bbbp_data()
    
    print(f"训练集大小: {len(train_data['atoms'])}")
    print(f"验证集大小: {len(valid_data['atoms'])}")
    print(f"测试集大小: {len(test_data['atoms'])}")
    
    # 准备训练数据格式
    # UniMol需要的格式: {'atoms': [...], 'coordinates': [...], 'target': [...]}
    train_data_formatted = {
        'atoms': train_data['atoms'],
        'coordinates': train_data['coordinates'],
        'target': [int(target[0]) for target in train_data['target']]  # 转换为标量
    }
    
    valid_data_formatted = {
        'atoms': valid_data['atoms'],
        'coordinates': valid_data['coordinates'],
        'target': [int(target[0]) for target in valid_data['target']]
    }
    
    test_data_formatted = {
        'atoms': test_data['atoms'],
        'coordinates': test_data['coordinates']
    }
    
    print("正在初始化UniMol训练器...")
    
    # 初始化训练器
    clf = MolTrain(
        task='classification',
        data_type='molecule',
        epochs=10,
        batch_size=16,
        metrics='auc',
        learning_rate=1e-4,
        save_path='./exp/bbbp_model',  # 保存模型的路径
        kfold=1
    )
    
    print("开始训练模型...")
    
    # 训练模型
    # 注意：这里我们传入训练数据，如果UniMol支持验证集，也可以传入
    try:
        clf.fit(data=train_data_formatted)
        print("模型训练完成!")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return
    
    print("正在准备预测器...")
    
    # 初始化预测器
    try:
        predictor = MolPredict(load_model='./exp/bbbp_model')
        print("开始在测试集上进行预测...")
        
        # 在测试集上进行预测
        test_results = predictor.predict(data=test_data_formatted)
        print("测试完成!")
        print(f"测试结果形状: {np.array(test_results).shape}")
        
        # 如果测试集有标签，计算准确率
        if 'target' in test_data and len(test_data['target']) > 0:
            test_labels = [int(target[0]) for target in test_data['target']]
            test_predictions = np.array(test_results)
            
            print(f"预测结果形状: {test_predictions.shape}")
            print(f"标签数量: {len(test_labels)}")
            
            # 处理预测结果，获取概率和类别
            if test_predictions.ndim > 1 and test_predictions.shape[1] > 1:
                # 多类输出，取正类概率
                probabilities = test_predictions[:, 1]  # 假设第二列是正类概率
                predicted_classes = np.argmax(test_predictions, axis=1)
            else:
                # 单一输出，可能是概率或logits
                if np.max(test_predictions) <= 1.0 and np.min(test_predictions) >= 0.0:
                    # 看起来像概率
                    probabilities = test_predictions.flatten()
                else:
                    # 看起来像logits，需要sigmoid转换
                    probabilities = 1 / (1 + np.exp(-test_predictions.flatten()))
                predicted_classes = (probabilities > 0.5).astype(int)
            
            # 计算各种指标
            accuracy = accuracy_score(test_labels, predicted_classes)
            roc_auc = roc_auc_score(test_labels, probabilities)
            
            print("=" * 50)
            print("测试集评估结果：")
            print("=" * 50)
            print(f"准确率 (Accuracy): {accuracy:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print("\n分类报告：")
            print(classification_report(test_labels, predicted_classes, 
                                      target_names=['非BBB穿透', 'BBB穿透']))
            
            # 显示一些统计信息
            print(f"\n数据统计：")
            print(f"正类样本数: {sum(test_labels)}")
            print(f"负类样本数: {len(test_labels) - sum(test_labels)}")
            print(f"正类比例: {sum(test_labels) / len(test_labels):.4f}")
            
            return {
                'predictions': test_results,
                'probabilities': probabilities,
                'labels': test_labels,
                'accuracy': accuracy,
                'roc_auc': roc_auc
            }
        
        return test_results
        
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        return None


def main():
    """
    主函数
    """
    print("=" * 50)
    print("BBBP数据集 UniMol 训练和测试")
    print("=" * 50)
    
    results = train_and_test_bbbp()
    
    if results is not None:
        if isinstance(results, dict) and 'roc_auc' in results:
            print("\n训练和测试流程完成!")
            print(f"最终结果总结：")
            print(f"- ROC-AUC: {results['roc_auc']:.4f}")
            print(f"- 准确率: {results['accuracy']:.4f}")
        else:
            print("训练和测试流程完成，但无法计算评估指标!")
    else:
        print("训练和测试过程中出现错误。")


if __name__ == "__main__":
    main()