import os
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def batch_evaluation(lst_file:str, 
pred_dir: str, 
verbose:bool=True
):
    # ground-truth items
    items=list()
    with open(lst_file,"r") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            _,label,rel_path=line.split()
            items.append((label,rel_path))
    # loading predictions and true lables
    y_true = []
    y_pred = []
    missing=[]
    for label, rel_path in items:
        pred_path=os.path.join(pred_dir,rel_path+".out")
        if not os.path.isfile(pred_path):
            missing.append(rel_path)
            continue
        with open(pred_path,"r") as f2:
            content=f2.read().strip()
            if not content:
                missing.append(rel_path)
                continue 
            prob_list=json.loads(content)
            probs = np.asarray(prob_list,dtype=float)
            y_pred.append(int(probs.argmax()))
            y_true.append(int(label))
    y_true = np.asarray(y_true,dtype=int)
    y_pred = np.asarray(y_pred,dtype=int)
    if verbose:

        print(f"Total lst entries: {len(items)}")
        print(f"Predictions found: {len(y_true)}")
        if missing:
            print(f"Missing predictions for {len(missing)} images (showing up to 10): {missing[:10]}")
    # generating reports
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["NORMAL","PNEUMONIA"])
    values_format=".2f"
    disp.plot(values_format=values_format, cmap="Blues")
    plt.title("Test data confusion matrix")
    
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    report=classification_report(
        y_true,
        y_pred,
        target_names=["NORMAL","PNEUMONIA"],
        digits=3,
        zero_division=0)
    return report
  
    

if __name__== "__main__":
    image_dir = "inferences"
    lst_path="dataset/chest_xray/test.lst"
    batch_evaluation(lst_path,image_dir)