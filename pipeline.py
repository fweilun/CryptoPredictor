from data import SingleUSStock, UnEmployRate
import factor
from typing import List


def main():
    selected_factor_cls:List[factor.BaseFactor] = [
        factor.weilun01,
        factor.weilun02,
        factor.weilun03, 
        factor.Cross, 
        factor.Slope,
        factor.Skewness
    ]
    
    for ftr_cls in selected_factor_cls:
        ftr_cls.cfg = factor.cfg[ftr_cls.__name__]
    
    selected_factors = []
    for ftr_cls in selected_factor_cls:
        for config in ftr_cls.cfg["signal_list"]:
            selected_factors.append(ftr_cls(**config["argument"]))
    
    [print(str(el)) for el in selected_factors]


if __name__ == "__main__":
    
    main()
    
    

