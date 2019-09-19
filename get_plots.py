import matplotlib.pyplot as plt
import pandas as pd

def get_precision_recall_curve(csv1, csv2):
    data1 = pd.read_csv(csv1)
    data2 = pd.read_csv(csv2)

    plt.plot(data1['recall'], data1['precision'], label='WIDER')
    plt.plot(data2['recall'], data2['precision'], label='Custom')
    plt.title("Custom Dataset vs WIDER dataset - WIDER Validation Set")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='lower left')

    #plt.show()
    plt.savefig("comparison_WIDER_val_52.png", bbox_inches='tight')

if __name__ == '__main__':
    wider_csv = '/home/giancarlo/Documents/Face_Detection_test/Precision-Recall/wider_val_wider.csv'
    custom_csv = '/home/giancarlo/Documents/Face_Detection_test/Precision-Recall/wider_val_custom_52.csv'
    get_precision_recall_curve(wider_csv, custom_csv)