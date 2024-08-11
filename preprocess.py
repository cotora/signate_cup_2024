import re
import kanjize
import pandas as pd
import unicodedata

def extract_age(s):
    result=30

    if pd.isna(s):
        return s

    kanji=["一","二","三","四","五","六","七","八","九","十"]

    pattern=r"\d+"
    search_result=re.search(pattern,s)

    if search_result:
        result=int(search_result.group())
    else:
        s_cp=""
        for c in list(s):
            if c in kanji:
                s_cp+=c
        result=int(kanjize.kanji2number(s_cp))

    if '代' in list(s):
        result+=5

    return result

def extract_duration(s):
    if pd.isna(s):
        return s  

    pattern=r"\d+"
    search_result=re.search(pattern,s)

    result=0

    if search_result:
        result=int(search_result.group())

    if "分" in s:
        result*=60

    return result

def extract_gender(s):
    if pd.isna(s):
        return s

    return unicodedata.normalize("NFKC",s).replace(" ","").lower()

def extract_product_pitched(s):
    if pd.isna(s):
        return s
    
    result=s
    
    result=unicodedata.normalize("NFKC",result).replace(" ","").lower()
    
    mapping={"μ":"m","е":"e","α":"a","×":"x","տ":"s","а":"a","ѵ":"v","ѕ":"s","ı":"i","|":"l","β":"b","в":"b","𐊡":"b","ς":"c","ᗞ":"d","ꓢ":"s","ꭰ":"d","ε":"e","ι":"i","ո":"n","с":"c"}
    for k,v in mapping.items():
        result=result.replace(k,v)

    return result

def extract_designation(s):
    if pd.isna(s):
        return s

    mapping={"μ":"m","е":"e","α":"a","×":"x","տ":"s","а":"a","ѵ":"v","ѕ":"s"}
    result=s
    for k,v in mapping.items():
        result=result.replace(k,v)

    result=unicodedata.normalize("NFKD",result).replace(" ","").lower()
    return result

def extract_monthly_income(s):

    if pd.isna(s):
        return s

    pattern=r'\d+\.?\d*'

    search_result=re.search(pattern,s)

    result=0

    if search_result:
        result=float(search_result.group())

    if "万" in s:
        result*=10000

    return result

def split_customer_info(s):

    result=unicodedata.normalize("NFKD",s)

    if "、" in result:
        result=result.replace("、"," ")
    if "/" in result:
        result=result.replace("/"," ")
    if "," in result:
        result=result.replace(","," ")

    result_split=result.split()

    return result_split

def extract_marriage_info(s):
    customer_info_split=split_customer_info(s)

    return customer_info_split[0]

def extract_car_info(s):
    customer_info_split=split_customer_info(s)

    car_info=customer_info_split[1]

    not_words=["なし","未"]

    for word in not_words:
        if word in car_info:
            return "yes"

    return "no"

def extract_child_info(s):
    customer_info_split=split_customer_info(s)

    child_info=customer_info_split[2]

    if len(customer_info_split)>3:
        child_info+=customer_info_split[3]
    
    pattern=r"\d+"

    search_result=re.search(pattern,child_info)

    if search_result:
        return int(search_result.group())
    else:
        strange_words=["わからない","不明","不詳"]
        for word in strange_words:
            if word in child_info:
                return None
            
        return 0


def preprocess(df):
    df["Age"]=df["Age"].apply(extract_age)
    df["DurationOfPitch"]=df["DurationOfPitch"].apply(extract_duration)
    df["Gender"]=df["Gender"].apply(extract_gender)
    df["ProductPitched"]=df["ProductPitched"].apply(extract_product_pitched)
    df["Designation"]=df["Designation"].apply(extract_designation)
    df["MonthlyIncome"]=df["MonthlyIncome"].apply(extract_monthly_income)
    df["MarriageStatus"]=df["customer_info"].apply(extract_marriage_info)
    df["CarOwnership"]=df["customer_info"].apply(extract_car_info)
    df["ChildNum"]=df["customer_info"].apply(extract_child_info)
    df=df.drop("customer_info",axis=1)
    df=df.fillna(df.mean(numeric_only=True))
    df=df.fillna(df.mode().iloc[0])
    df=pd.get_dummies(df)

    return df

if __name__=="__main__":
    pass