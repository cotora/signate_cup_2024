import re
import kanjize
import pandas as pd
import unicodedata
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import numpy as np


def extract_age(s):
    result = 0

    if pd.isna(s):
        return s
    
    s=unicodedata.normalize("NFKD", s)

    kanji = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

    pattern = r"\d+"
    search_result = re.search(pattern, s)

    if search_result:
        result = int(search_result.group())
    else:
        s_cp = ""
        for c in list(s):
            if c in kanji:
                s_cp += c
        result = int(kanjize.kanji2number(s_cp))

    if "代" in list(s):
        result += 5

    return int(result)


def extract_duration(s):
    if pd.isna(s):
        return s

    pattern = r"\d+"
    search_result = re.search(pattern, s)

    result = 0

    if search_result:
        result = int(search_result.group())

    if "分" in s:
        result *= 60

    return result


def extract_gender(s):
    if pd.isna(s):
        return s

    return unicodedata.normalize("NFKC", s).replace(" ", "").lower()


def extract_product_pitched(s):
    if pd.isna(s):
        return s

    result = s

    result = unicodedata.normalize("NFKC", result).replace(" ", "").lower()

    mapping = {
        "μ": "m",
        "е": "e",
        "α": "a",
        "×": "x",
        "տ": "s",
        "а": "a",
        "ѵ": "v",
        "ѕ": "s",
        "ı": "i",
        "|": "l",
        "β": "b",
        "в": "b",
        "𐊡": "b",
        "ς": "c",
        "ᗞ": "d",
        "ꓢ": "s",
        "ꭰ": "d",
        "ε": "e",
        "ι": "i",
        "ո": "n",
        "с": "c",
    }
    for k, v in mapping.items():
        result = result.replace(k, v)

    return result


def extract_designation(s):
    if pd.isna(s):
        return s
    
    result = s
    result = unicodedata.normalize("NFKC", result).replace(" ", "").lower()

    mapping = {
        "μ": "m",
        "е": "e",
        "α": "a",
        "×": "x",
        "տ": "s",
        "а": "a",
        "ѵ": "v",
        "ѕ": "s",
    }

    for k, v in mapping.items():
        result = result.replace(k, v)

    return result


def extract_monthly_income(s):

    if pd.isna(s):
        return s

    pattern = r"\d+\.?\d*"

    search_result = re.search(pattern, s)

    result = 0

    if search_result:
        result = float(search_result.group())

    if "万" in s:
        result *= 10000

    return result


def split_customer_info(s):

    result = unicodedata.normalize("NFKD", s)

    if "、" in result:
        result = result.replace("、", " ")
    if "/" in result:
        result = result.replace("/", " ")
    if "," in result:
        result = result.replace(",", " ")

    result_split = result.split()

    return result_split


def extract_marriage_info(s):
    customer_info_split = split_customer_info(s)

    return customer_info_split[0]


def extract_car_info(s):
    customer_info_split = split_customer_info(s)

    car_info = customer_info_split[1]

    not_words = ["なし", "未"]

    for word in not_words:
        if word in car_info:
            return "yes"

    return "no"


def extract_child_info(s):
    customer_info_split = split_customer_info(s)

    child_info = customer_info_split[2]

    if len(customer_info_split) > 3:
        child_info += customer_info_split[3]

    pattern = r"\d+"

    search_result = re.search(pattern, child_info)

    if search_result:
        return int(search_result.group())
    else:
        strange_words = ["わからない", "不明", "不詳"]
        for word in strange_words:
            if word in child_info:
                return np.nan

        return 0
    
def extract_number_of_trips(s):
    if pd.isna(s):
        return s

    pattern = r"\d+"

    search_result = re.search(pattern, s)

    result = 0

    if search_result:
        result = int(search_result.group())

    if "半年" in s:
        result *= 2
    elif "四半期" in s:
        result *= 4

    return result

def extract_type_of_contacts(s):
    if pd.isna(s):
        return s

    return s.replace(" ", "_")

def extract_number_of_followups(num):
    if pd.isna(num):
        return num

    result = num

    if result>=100:
        result/=100

    return result


def target_encoder(train, test, target_col, cat_cols):
    for col in cat_cols:
        # テストデータをエンコード
        target_mean = train.groupby(col)[target_col].mean()
        test.loc[:,col] = test[col].map(target_mean)
        test[col] = test[col].astype(float)

        tmp = np.repeat(np.nan, train.shape[0])
    
        # 学習データを分割
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, va_idx in kf.split(train):
            tr_x, va_x = train.iloc[tr_idx], train.iloc[va_idx]
            target_mean = tr_x.groupby(col)[target_col].mean()
            tmp[va_idx] = va_x[col].map(target_mean)

        train.loc[:,col] = tmp
        train[col] = train[col].astype(float)

    return train, test


def preprocess(df):
    cat_features=["Occupation","Gender","ProductPitched","Passport","Designation","MarriageStatus","CarOwnership","TypeofContact"]

    df["Age"] = df["Age"].apply(extract_age)
    df["DurationOfPitch"] = df["DurationOfPitch"].apply(extract_duration)
    df["Gender"] = df["Gender"].apply(extract_gender)
    df["ProductPitched"] = df["ProductPitched"].apply(extract_product_pitched)
    df["Designation"] = df["Designation"].apply(extract_designation)
    df["MonthlyIncome"] = df["MonthlyIncome"].apply(extract_monthly_income)
    df["MarriageStatus"] = df["customer_info"].apply(extract_marriage_info)
    df["CarOwnership"] = df["customer_info"].apply(extract_car_info)
    df["ChildNum"] = df["customer_info"].apply(extract_child_info)
    df["NumberOfTrips"] = df["NumberOfTrips"].apply(extract_number_of_trips)
    df["TypeofContact"] = df["TypeofContact"].apply(extract_type_of_contacts)
    df["Occupation"] = df["Occupation"].apply(extract_type_of_contacts)
    df["NumberOfFollowups"] = df["NumberOfFollowups"].apply(extract_number_of_followups)
    df = df.drop("customer_info", axis=1)

    for col in cat_features:
        df[col] = df[col].astype(str).fillna("nan")

    #df["ChildRatio"]=df["ChildNum"]/df["NumberOfPersonVisiting"]
    #df = df.fillna(df.mean(numeric_only=True))
    #df = df.fillna(df.mode().iloc[0])

    """
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    """
    

    #df = pd.get_dummies(df)

    return df


if __name__ == "__main__":
    train_data=pd.read_csv("data/input/train.csv")
    test_data=pd.read_csv("data/input/test.csv")
    all_data = pd.concat([train_data, test_data])
    all_data = preprocess(all_data)

    for col in all_data.columns:
        print(all_data[col].value_counts())
        print()
