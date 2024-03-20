import numpy as np
import matplotlib.pyplot as plt

def plot_label_accuracies(true_labels, predicted_labels, title):
    predicted_labels = np.array(predicted_labels)
    
    unique_labels = np.unique(true_labels)
    
    plt.figure(figsize=(10, 7))
    
    overall_accuracy = np.mean(predicted_labels == true_labels) * 100

    for i, label in enumerate(unique_labels):
        label_indices = np.where(true_labels == label)[0]
        
        label_predicted_labels = predicted_labels[label_indices]
        
        label_accuracy = np.mean(label_predicted_labels == label) * 100
        
        bar = plt.bar(str(label), label_accuracy, color='skyblue')
        
        plt.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height(), f'{label_accuracy:.2f}%', 
                 ha='center', va='bottom')

    plt.text(0.95, 0.95, f'Overall Accuracy: {overall_accuracy:.2f}%', transform=plt.gca().transAxes, 
             ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.title(title)
    plt.xlabel('True Label')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)  # Optionally set limits to make the scale from 0% to 100%
    plt.yticks(np.arange(0, 101, 10), [f'{i}%' for i in range(0, 101, 10)])
    plt.show()