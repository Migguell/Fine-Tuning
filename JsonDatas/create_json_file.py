import json  
from JsonDatas.datas import datas

with open("data.json", "w") as f:  
    json.dump(datas, f, indent=4)  

print("successfully created")