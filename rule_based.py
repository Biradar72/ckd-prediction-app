import pandas as pd

def rule_based_ckd_predict(data):
    """
    Predict CKD based on WHO-style logic.
    Input: dict or pandas Series
    Output: string result
    """
    bp = data.get('bp', 0)
    sg = data.get('sg', 1.015)
    al = data.get('al', 0)
    sc = data.get('sc', 1.2)
    bu = data.get('bu', 30)
    hemo = data.get('hemo', 13)
    pc = data.get('pc', 0)
    rbc = data.get('rbc', 0)
    wc = data.get('wc', 8000)
    appet = data.get('appet', 1)
    htn = data.get('htn', 0)

    
    if bp > 140 and sg < 1.010 and al >= 2:
        return "High Risk of CKD"
    elif hemo < 10 and pc == 1:
        return "Moderate Risk of CKD"
    elif rbc == 1 and sc > 1.5:
        return "Possible CKD"
    elif bu > 100 or wc > 20000:
        return "Risk of CKD (needs further tests)"
    elif appet == 0 and htn == 1:
        return "Early Risk of CKD"
    else:
        return "Low Risk / No CKD"
