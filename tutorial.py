# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="7dc8dd76"
# # AirPodsを活用した顎関節症判定チャレンジ

# %% [markdown] id="b47bbdc9"
# これは「AirPodsを活用した顎関節症判定チャレンジ」の分析・モデリングチュートリアルである. 分析・モデリングをする際の参考にされたい.

# %% [markdown] id="a9199988" jp-MarkdownHeadingCollapsed=true
# ## 前準備

# %% [markdown] id="93728373" jp-MarkdownHeadingCollapsed=true
# ### データの準備

# %% [markdown] id="b555242e"
# 配布されている`train.zip`, `test.zip`をダウンロードし, このノートブックと同じディレクトリに配置して, zipファイルは解凍する. 解凍後以下のようなディレクトリができていることを確認.
#
# ```
# .
# ├── train              # 学習用データ
# │   ├── negative
# │   └── positive
# ├── test               # 評価用データ
# │   ├── 000
# │   └── ...
# └── tutorial.ipynb     # このノートブックファイル
# ```
#
# 各データの定義については配布されている`README.pdf`を参照すること.

# %% colab={"base_uri": "https://localhost:8080/"} id="ZzUwgkbArO8Z" outputId="25189f20-7684-40e4-8c0e-ba97353d986d"
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="acd9ff32"
# #### Google Colaboratoryを使う場合

# %% [markdown] id="0a5df07b"
# 自身のドライブなどにこのjupyternotebookファイルをアップロードして立ち上げることでGoogle Colaboratoryを起動し, `/content`以下に`train.zip`, `test.zip`をアップロードする. その後下記コマンドを実行することで, zipファイルは解凍される.

# %% colab={"base_uri": "https://localhost:8080/"} id="f905e190" outputId="2e63f124-7d99-4706-fe78-926332aa2b53"
# !unzip "/content/drive/MyDrive/YIL-challenge-2/test.zip" -d "/content"
# !unzip "/content/drive/MyDrive/YIL-challenge-2/train.zip"  -d "/content"


# %% [markdown] id="50321c39" jp-MarkdownHeadingCollapsed=true
# ### ライブラリのインストール

# %% [markdown] id="cf22da5a"
# これから行う分析やモデリングに必要なライブラリをインストールする. 主に必要なライブラリは以下の通り.
#
# - pandas
# - numpy
# - torch
# - scipy
# - scikit-learn
# - matplotlib
# - skorch
#
# インストールされていないなら下記コマンドでインストールすること.

# %% colab={"base_uri": "https://localhost:8080/"} id="c53d6c02" outputId="829cb2fa-8178-49c5-c008-dc534f04d031"
# ! pip install aeon scikit-learn pandas matplotlib

# %% [markdown] id="ef6dd194" jp-MarkdownHeadingCollapsed=true
# ## ライブラリのインポートと設定

# %% [markdown] id="46c21ccd"
# 今後の作業で必要になる「道具」（＝Pythonのライブラリ）を用意する. また, データやファイルを保存しているフォルダの場所（ルートディレクトリ）を決めておく. これを最初にやっておくと, 今後データを読み込んだり保存したりするときに, 毎回場所を指定しなくてすむので便利である.
#
# 下記セルのコメントアウトにてインポートしているライブラリの役割を簡単に確認されたい.

# %% id="df6c47a8"
# データ分析や機械学習に使う道具（ライブラリ）を準備します

import os                          # ファイルやフォルダの操作を行う標準ライブラリ
import pandas as pd                # 表形式データ（DataFrame）を扱う。データの読み書きや加工が得意
import numpy as np                 # 高速な数値計算や行列・配列操作に使う
import matplotlib.pyplot as plt    # グラフや可視化を行う

from scipy.signal import resample  # 信号処理で使うリサンプリング（データ数変更）機能を利用可能

from sklearn.model_selection import train_test_split  # データを訓練用・テスト用に分割する関数
from sklearn.metrics import roc_curve, auc, roc_auc_score  # ROC曲線とAUCスコア（評価指標）計算に使う

import torch                       # PyTorch本体
import torch.nn as nn              # ニューラルネットワークの各種層
import torch.nn.functional as F    # 活性化関数など
from torch.utils.data import DataLoader, TensorDataset  # データローダー

import pickle  # モデルの保存・読み込み用


# データの読み込み場所, 結果の保存場所を決めておきます
DATA_DIR = '.'     # データファイルがあるフォルダ（ここではカレントフォルダを指定. Google Colaboratoryを使っていてGoogle Driveをマウントしてデータを展開しているなら`/content/drive/MyDrive`などと設定）
OUTPUT_DIR = '.'   # 結果などを保存するフォルダ（ここではカレントフォルダを指定. Google Colaboratoryを使っていてGoogle Driveをマウントしてデータを展開しているなら`/content/drive/MyDrive`などと設定）

# %% [markdown] id="fa93b155" jp-MarkdownHeadingCollapsed=true
# ## データの整理

# %% [markdown] id="1884b431"
# これから分析に使うためのデータファイル（主にCSVファイル）の情報を整理して, いろいろな分析をできるようにする.

# %% [markdown] id="b0899719" jp-MarkdownHeadingCollapsed=true
# ### 学習用

# %% [markdown] id="4523470f"
# 学習用として与えられているのは`./train`以下にある被験者ごとのairpodsより取得した系列データである. 以下の処理を行う.
#
# - 分類で使うラベル名("positive")とファイルパス("fpath"), 被験者ID("person"), セット数("set_id"), 動作の種類("move"), 左右どちらか("leftorright")を対応させて定義する.
#     - ラベル名は"顎関節症の自覚症状がある"=1, "顎関節症の自覚症状がない"=0
#     - 動作の種類は"自力最大開口"='01', "側転運動"='02'
#     - 左右どちらかは'left'=左耳から取得, 'right'=右耳から取得
# - 被験者の数やセット回数などを確認する.
#
# これらは分析や可視化を行うための準備である.

# %% colab={"base_uri": "https://localhost:8080/"} id="923755bb" outputId="2f6e4411-cc55-4861-a5aa-39ccb4e2ecfa"
# データファイルの場所（ディレクトリ名：train）を指定します
data_path = os.path.join(DATA_DIR, 'train')   # DATA_DIRは事前にどこかで定義されている前提

train_master = []    # メタデータ（ファイルパスや属性情報）をリストで保持

# trainフォルダ以下に保存された全データ（階層構造）を探索
for label_name in os.listdir(data_path):  # label_name: ラベル（'positive'または'negative'のフォルダ）
    for person_id in os.listdir(os.path.join(data_path, label_name)):  # person_id: 被験者ID（サブディレクトリ）
        for file_name in os.listdir(os.path.join(data_path, label_name, person_id)):  # file_name: 実際のデータファイル名
            # ファイル名を_で区切って各属性を抽出（例: "set01_01_left.csv" なら set_id="01", move="01", leftorright="left"）
            set_id, move, leftorright = file_name.split('.')[0].split('_')

            # 各データについて、ファイルパスや被験者IDなどの情報をまとめて辞書にする
            train_master.append({
                'fpath': os.path.join(data_path, label_name, person_id, file_name),   # ファイルのフルパス
                'person': person_id,                                                  # 被験者ID
                'set_id': set_id,                                                     # セットID（同じ動作/回の区別）
                'move': move,                                                         # 動作名
                'leftorright': leftorright,                                           # 左右情報
                'positive': 1 if label_name == 'positive' else 0                      # ラベル（positive=1, negative=0）
            })

# リスト（train_master）をpandasのDataFrameに変換
train_master = pd.DataFrame(train_master)

# DataFrameの最初の数行を表示して、正しくデータを作れているか確認
print(train_master.head())

# 重複を除いた被験者（person）数を表示
print('\n被験者の数:', len(train_master['person'].unique()))

# 各被験者ごとに持っているセットIDの数を集計し、平均値を表示
print(
    '被験者ごとのセット回数:',
    train_master.groupby('person').apply(
        lambda x: len(x['set_id'].unique()), include_groups=False # type: ignore
    ).describe()['mean']   # type: ignore
)

# %% [markdown] id="ee4f2c7f" jp-MarkdownHeadingCollapsed=true
# ### 評価用

# %% [markdown] id="c3507879"
# 評価用として与えられているのは`./test`以下にあるairpodsの系列データのみである. 各被験者, セット回数(具体的に与えられるわけではない)ごとに2種類の動作(自力最大開口, 側方運動)と左右の組み合わせで計4パターンの測定結果がcsvファイルとして格納されている. なお, 学習用とは別の被験者のデータであることに注意. ラベル情報は与えられず, このラベルを当てることが今回の課題の主目的である. ここでは学習用データと同様に以下の処理を行う.
#
# - 評価用データは被験者, セット回数(ディレクトリ名としてIDが振られている)ごとにまとめられているので, それぞれの対応するデータのファイルパス("fpath")を紐づける.
#     - 各IDごとに4つの系列データファイルが紐づくことになる.
# - 推論の対象となるサンプル数を確認する. 各IDごとに推論を行うことになる.
#
# これらは学習した機械学習モデルによって評価用データに対して推論を行って, 結果を作成するための準備である.

# %% colab={"base_uri": "https://localhost:8080/"} id="7c3928e9" outputId="dc5816bb-8061-4f33-d43d-53a52fcb879c"
# テストデータのファイル情報を記録するリスト
test_master = []

# testディレクトリ配下の各サブディレクトリ（ID単位）を探索
for test_id in os.listdir(os.path.join(DATA_DIR, 'test')):  # test_id: テストID
    for file_name in os.listdir(os.path.join(DATA_DIR, 'test', test_id)):  # file_name: データファイル名
        # 各ファイルごとに「id」とファイルのフルパスを記録
        test_master.append({
            'id': test_id,  # サンプルID（被験者IDなど）
            'fpath': os.path.join(DATA_DIR, 'test', test_id, file_name)  # ファイルのフルパス
        })

# リストをDataFrameに変換して管理しやすくする
test_master = pd.DataFrame(test_master)

# データフレームの先頭5行を表示（正しく格納されているか確認）
print(test_master.head())

# テストデータのID（重複なし）の数を表示
print('\nサンプル数:', len(test_master['id'].unique()))

# %% colab={"base_uri": "https://localhost:8080/"} id="pxj74RBL1Wb3" outputId="50a8f728-b2d2-4fcc-b98c-5fe38aca4271"
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, ttest_ind
from scipy.signal import find_peaks
from scipy.fft import rfft
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

# ======================
# 対象チャネル: 全チャネルを使用
# ======================

# (CSVの全カラムを自動的に使用)


# ======================
# 特徴抽出関数
# ======================

def compute_features(signal):

    feats = {}
    x = signal
    n = len(x)
    dx = np.diff(x)
    ddx = np.diff(x, n=2)

    # 長さ非依存の統計量
    feats['mean'] = np.mean(x)
    feats['std'] = np.std(x)
    feats['p5'] = np.percentile(x, 5)      # min の代替（長さ依存しない）
    feats['p95'] = np.percentile(x, 95)     # max の代替
    feats['ptp'] = np.ptp(x)
    feats['skew'] = skew(x)
    feats['kurt'] = kurtosis(x)
    feats['rms'] = np.sqrt(np.mean(x**2))
    feats['tv'] = np.mean(np.abs(dx))       # sum → mean に変更
    feats['jerk_std'] = np.std(ddx) if len(ddx)>0 else 0

    peaks,_ = find_peaks(np.abs(x))
    feats['n_peaks'] = len(peaks) / n       # 長さで正規化

    feats['energy'] = np.mean(x**2)         # sum → mean に変更

    fft_vals = np.abs(rfft(x))
    n_fft = len(fft_vals)
    feats['fft_low'] = np.mean(fft_vals[:5]) if n_fft >= 5 else 0     # sum → mean
    feats['fft_mid'] = np.mean(fft_vals[5:15]) if n_fft >= 15 else 0  # sum → mean
    feats['fft_high'] = np.mean(fft_vals[15:]) if n_fft > 15 else 0   # sum → mean

    return feats


# ======================
# 全特徴生成 (左右差 + 相関 + 比率 + 個別チャネル)
# ======================

all_features = []
labels = []
persons = []

for (person, set_id, positive), group in train_master.groupby(['person','set_id','positive']):

    row_feats = {}

    for move in ['01','02']:

        rows = group[group['move']==move]

        left = None
        right = None

        for _, r in rows.iterrows():
            df = pd.read_csv(r['fpath']).set_index('Timestamp')
            df = df[~df.index.duplicated()].sort_index()

            if r['leftorright']=='left':
                left = df
            else:
                right = df

        if left is None or right is None:
            continue

        # 共通カラムのみ使用
        common_cols = [c for c in left.columns if c in right.columns]
        left = left[common_cols]
        right = right[common_cols]

        n = min(len(left), len(right))
        left = left.iloc[:n]
        right = right.iloc[:n]

        for i, col in enumerate(common_cols):
            l_vals = left.iloc[:, i].values
            r_vals = right.iloc[:, i].values

            # ① 左右差の統計量
            diff = l_vals - r_vals
            feats = compute_features(diff)
            for k, v in feats.items():
                row_feats[f'{move}_{col}_diff_{k}'] = v

            # ② 左右の相関係数
            if len(l_vals) > 2 and np.std(l_vals) > 0 and np.std(r_vals) > 0:
                corr = np.corrcoef(l_vals, r_vals)[0, 1]
            else:
                corr = 0.0
            row_feats[f'{move}_{col}_lr_corr'] = corr

            # ③ 左右の比率 (比率の統計量)
            ratio = l_vals / (np.abs(r_vals) + 1e-8)
            row_feats[f'{move}_{col}_ratio_mean'] = np.mean(ratio)
            row_feats[f'{move}_{col}_ratio_std'] = np.std(ratio)

            # ④ 左チャネル単体の統計量
            l_feats = compute_features(l_vals)
            for k, v in l_feats.items():
                row_feats[f'{move}_{col}_left_{k}'] = v

            # ⑤ 右チャネル単体の統計量
            r_feats = compute_features(r_vals)
            for k, v in r_feats.items():
                row_feats[f'{move}_{col}_right_{k}'] = v

    if len(row_feats)>0:
        all_features.append(row_feats)
        labels.append(positive)
        persons.append(person)


X = pd.DataFrame(all_features)
y = np.array(labels)
groups = np.array(persons)

print(f"特徴量数: {X.shape[1]}, サンプル数: {X.shape[0]}, 被験者数: {len(np.unique(groups))}")


# ======================
# 統計検定 (被験者単位 + FDR補正)
# ======================

# 被験者単位で集約 (独立性を確保)
X_person = X.copy()
X_person['person'] = groups
X_person['label'] = y
X_person = X_person.groupby('person').agg(
    {col: 'mean' for col in X.columns} | {'label': 'first'}
)

y_person = X_person['label'].values
feature_columns = [c for c in X_person.columns if c != 'label']

results = []

for col in feature_columns:

    pos = X_person[y_person==1][col].dropna()
    neg = X_person[y_person==0][col].dropna()

    if len(pos)<3 or len(neg)<3:
        continue

    t, p = ttest_ind(pos, neg, equal_var=False)
    d = (pos.mean()-neg.mean()) / np.sqrt((pos.var()+neg.var())/2)

    results.append([col, p, d])

results_df = pd.DataFrame(results, columns=['feature','p_raw','cohen_d'])

# FDR補正 (Benjamini-Hochberg法)
rejected, p_corrected, _, _ = multipletests(results_df['p_raw'], method='fdr_bh')
results_df['p_fdr'] = p_corrected
results_df['significant'] = rejected

# StratifiedGroupKFold AUC (リークなし)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
auc_results = []

for col in feature_columns:
    fold_aucs = []
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        test_labels = y[test_idx]
        if len(np.unique(test_labels)) < 2:
            continue
        auc = roc_auc_score(test_labels, X.iloc[test_idx][col])
        fold_aucs.append(auc)
    if len(fold_aucs)>0:
        auc_results.append([col, np.mean(fold_aucs)])

auc_df = pd.DataFrame(auc_results, columns=['feature','cv_auc'])

# 結合してランキング
results_df = results_df.merge(auc_df, on='feature', how='left')
results_df = results_df.sort_values('cv_auc', ascending=False)

print("\n=== 特徴量ランキング (被験者単位t検定 + FDR補正 + CV-AUC) ===")
print(results_df.head(20).to_string(index=False))


# %% colab={"base_uri": "https://localhost:8080/"} id="fwd_selection"
# ======================
# Forward Feature Selection (貪欲探索)
# ======================

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

# === CV方式の選択 ===
# 'LOPO' : Leave-One-Person-Out (21 folds, 再現性あり, 保守的)
# 'RSKF' : Repeated StratifiedGroupKFold (10 seeds × 5 folds)
CV_MODE = 'LOPO'

# === 正則化パラメータ ===
# C が小さいほど正則化が強い (デフォルト: 1.0)
REG_C = 0.05

def cv_auc(X_mat, y_vec, groups_vec):
    """CV_MODEに応じたAUCを計算"""
    if CV_MODE == 'LOPO':
        logo = LeaveOneGroupOut()
        all_probs = np.zeros(len(y_vec))
        for train_idx, test_idx in logo.split(X_mat, y_vec, groups=groups_vec):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_mat[train_idx])
            X_te = scaler.transform(X_mat[test_idx])
            model = LogisticRegression(C=REG_C)
            model.fit(X_tr, y_vec[train_idx])
            all_probs[test_idx] = model.predict_proba(X_te)[:, 1]
        return roc_auc_score(y_vec, all_probs)
    else:  # RSKF
        seed_means = []
        for seed in range(42, 52):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_aucs = []
            for train_idx, test_idx in cv.split(X_mat, y_vec, groups=groups_vec):
                y_test = y_vec[test_idx]
                if len(np.unique(y_test)) < 2:
                    continue
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_mat[train_idx])
                X_te = scaler.transform(X_mat[test_idx])
                model = LogisticRegression(C=REG_C)
                model.fit(X_tr, y_vec[train_idx])
                prob = model.predict_proba(X_te)[:, 1]
                fold_aucs.append(roc_auc_score(y_test, prob))
            if fold_aucs:
                seed_means.append(np.mean(fold_aucs))
        return np.mean(seed_means) if seed_means else 0.0

print(f"CV方式: {CV_MODE}")

# 候補: ランキング上位50個に絞る (計算時間削減)
TOP_K = 50
candidates = results_df.head(TOP_K)['feature'].tolist()

y_arr = np.asarray(y)
g_arr = np.asarray(groups)

selected = []
best_score = 0.0
MAX_FEATURES = 10  # 最大特徴数

print(f"候補特徴量: {len(candidates)}個 → 最大{MAX_FEATURES}個を選択\n")

for step in range(MAX_FEATURES):
    best_feat = None
    best_new_score = best_score

    for feat in candidates:
        if feat in selected:
            continue

        trial = selected + [feat]
        X_trial = X[trial].values
        score = cv_auc(X_trial, y_arr, g_arr)

        if score > best_new_score:
            best_new_score = score
            best_feat = feat

    if best_feat is None:
        print(f"\nStep {step+1}: 改善する特徴なし → 終了")
        break

    selected.append(best_feat)
    best_score = best_new_score
    print(f"Step {step+1}: + {best_feat}  → AUC = {best_score:.4f}")

print(f"\n=== 選択された特徴量 ({len(selected)}個) ===")
for f in selected:
    print(f"  - {f}")
print(f"最終 AUC: {best_score:.4f}")

FEATURE_NAMES = selected


# %% colab={"base_uri": "https://localhost:8080/"} id="YGvOzIP64Jn8" outputId="b8929fad-d7e4-45e3-fe26-7f0d48703ecb"
# ======================
# CV 評価 (選択された特徴量)
# ======================

print(f"使用特徴量: {len(FEATURE_NAMES)}個")
for f in FEATURE_NAMES:
    print(f"  - {f}")

X_cv = X[FEATURE_NAMES].values
y_cv = np.asarray(y)
groups_cv = np.asarray(groups)

if CV_MODE == 'LOPO':
    print(f"\n--- LOPO CV (21 folds) ---")
    logo = LeaveOneGroupOut()
    lopo_probs = np.zeros(len(y_cv))

    for train_idx, test_idx in logo.split(X_cv, y_cv, groups=groups_cv):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]
        person = groups_cv[test_idx][0]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(C=REG_C)
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        lopo_probs[test_idx] = prob

        label_str = "pos" if y_test[0] == 1 else "neg"
        print(f"  Person {person} ({label_str}): mean prob = {np.mean(prob):.4f}")

    lopo_auc = roc_auc_score(y_cv, lopo_probs)
    print(f"\n=== LOPO AUC: {lopo_auc:.4f} ===")

else:
    print(f"\n--- Repeated StratifiedGroupKFold ---")
    SEEDS = list(range(42, 52))
    all_aucs = []

    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        seed_aucs = []

        for train_idx, test_idx in cv.split(X_cv, y_cv, groups=groups_cv):
            X_train, X_test = X_cv[train_idx], X_cv[test_idx]
            y_train, y_test = y_cv[train_idx], y_cv[test_idx]

            if len(np.unique(y_test)) < 2:
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = LogisticRegression(C=REG_C)
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            seed_aucs.append(auc)

        if len(seed_aucs) > 0:
            seed_mean = np.mean(seed_aucs)
            all_aucs.append(seed_mean)
            fold_str = ", ".join([f"{a:.3f}" for a in seed_aucs])
            print(f"  Seed {seed}: Mean AUC = {seed_mean:.4f}  [{fold_str}]  ({len(seed_aucs)} folds)")

    print(f"\n=== Repeated CV ({len(all_aucs)} valid seeds) ===")
    print(f"Mean AUC: {np.mean(all_aucs):.4f} (±{np.std(all_aucs):.4f})")
    print(f"Min: {np.min(all_aucs):.4f}, Max: {np.max(all_aucs):.4f}")


# %% colab={"base_uri": "https://localhost:8080/"} id="gCpvtWrN5OHe" outputId="4ac6a8bb-f813-4d32-d612-42cba80cc87a"
# ======================
# 提出用: 全データで学習 → テスト予測
# ======================

TEST_ROOT = os.path.join(DATA_DIR, 'test')

# --- 学習 (set単位で全データ使用) ---
X_train_final = X[FEATURE_NAMES].values
y_train_final = np.asarray(y)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_final)

model_final = LogisticRegression(C=REG_C)
model_final.fit(X_train_scaled, y_train_final)

print(f"学習完了: {len(X_train_final)}サンプル (set単位), 特徴量: {len(FEATURE_NAMES)}個")


# --- テスト特徴抽出 ---

def extract_test_features(folder_path, feature_names):
    """テストデータから特徴を抽出 (学習と同じ5種類)"""
    row_feats = {}

    for move in ['01', '02']:
        left_path = os.path.join(folder_path, f"{move}_left.csv")
        right_path = os.path.join(folder_path, f"{move}_right.csv")

        left = pd.read_csv(left_path).set_index("Timestamp")
        right = pd.read_csv(right_path).set_index("Timestamp")

        left = left[~left.index.duplicated()].sort_index()
        right = right[~right.index.duplicated()].sort_index()

        common_cols = [c for c in left.columns if c in right.columns]
        left = left[common_cols]
        right = right[common_cols]

        n = min(len(left), len(right))
        left = left.iloc[:n]
        right = right.iloc[:n]

        for i, col in enumerate(common_cols):
            l_vals = left.iloc[:, i].values
            r_vals = right.iloc[:, i].values

            # ① 左右差
            diff = l_vals - r_vals
            feats = compute_features(diff)
            for k, v in feats.items():
                row_feats[f'{move}_{col}_diff_{k}'] = v

            # ② 相関
            if len(l_vals) > 2 and np.std(l_vals) > 0 and np.std(r_vals) > 0:
                corr = np.corrcoef(l_vals, r_vals)[0, 1]
            else:
                corr = 0.0
            row_feats[f'{move}_{col}_lr_corr'] = corr

            # ③ 比率
            ratio = l_vals / (np.abs(r_vals) + 1e-8)
            row_feats[f'{move}_{col}_ratio_mean'] = np.mean(ratio)
            row_feats[f'{move}_{col}_ratio_std'] = np.std(ratio)

            # ④ 左単体
            l_feats = compute_features(l_vals)
            for k, v in l_feats.items():
                row_feats[f'{move}_{col}_left_{k}'] = v

            # ⑤ 右単体
            r_feats = compute_features(r_vals)
            for k, v in r_feats.items():
                row_feats[f'{move}_{col}_right_{k}'] = v

    return [row_feats[f] for f in feature_names]


# --- テスト予測 ---
test_ids = sorted(os.listdir(TEST_ROOT))
results = []

for tid in test_ids:
    folder = os.path.join(TEST_ROOT, tid)
    feat_values = extract_test_features(folder, FEATURE_NAMES)
    feat_scaled = scaler_final.transform([feat_values])
    prob = model_final.predict_proba(feat_scaled)[0, 1]
    results.append({"id": tid, "positive": prob})

submission = pd.DataFrame(results)
submission = submission.sort_values('id').reset_index(drop=True)
submission.to_csv("submission.csv", index=False, header=False)

print(f"\n✓ submission.csv created ({len(submission)} samples)")
print(f"  ID range: {submission['id'].iloc[0]} → {submission['id'].iloc[-1]}")
print(f"  Prediction range: [{submission['positive'].min():.4f}, {submission['positive'].max():.4f}]")


# %% [markdown] id="dbb50005" jp-MarkdownHeadingCollapsed=true
# ## 深層学習の実装時によくある注意点・Tips

# %% [markdown] id="85f2b7df"
# 主な工夫点は下記の通り. 実装にチャレンジして精度改善できるか試されたい.
#
# 1. 特徴量拡張・データ拡張
#     - **追加特徴量の導入**  
#         - 加速度・角速度・クォータニオンなどを「差分」や「移動平均」「標準偏差」「エネルギー」などの統計量でも特徴量化
#         - 複数ウィンドウ幅での時系列要約・小さな区間の最大・最小・ピーク回数などを追加
#     - **ノイズ付与やシャッフルによるデータ拡張**  
#         - トレーニング時にランダムな小ノイズ/ランダム時刻シフト/部分欠損を加えて学習データを水増し（データ拡張）
#         - データミラー/左右反転が妥当なら適用する
# 1. モデルアーキテクチャの改良
#     - **Conv1D層の深層化やカーネル幅・チャンネル数の調整**
#         - チャンネル数やフィルター幅を雑に数パターン試し, 最も安定するものを採用
#     - **畳み込み＋GRUやLSTMなどのRNN併用ハイブリッド**
#         - Conv後にGRU/LSTMを付加し「時系列での長い依存関係」も捉える
#     - **Attention層やSelf-Attention (Transformer的小規模モジュール) の組み込み**
#         - センサーデータの重要な時刻点を自動で強調
#     - **DropoutやLayerNorm, GELU等の正則化/活性化工夫**
# 1. 前処理・入力の工夫
#     - **グローバルな標準化・正規化**
#         - サンプルごとの標準化から「学習用全体の平均・分散で標準化」への切り替え検討（一般化性能向上）.
#     - **入力ウィンドウ長や次元数の最適化**
#         - `seq_len`を増減して最適値探索.
#     - **欠損補完や異常値の明示的処理**
# 1. 損失関数・評価・学習プロセス工夫
#     - **ロス関数を工夫する**（クラス不均衡ならFocal Loss, Weighted Lossの活用）
#     - **アーリーストッピング/EarlyStopping導入**（バリデーションロスやAUC見て自動で学習終了）
#     - **k-foldクロスバリデーションによる精度安定化評価**
#     - **LearningRateSchedulerの導入で動的に学習率調整**
# 1. アンサンブル・他手法併用
#     - **同じ形式でLightGBMやCatBoost等の古典機械学習で特徴量を混ぜる**
#     - **複数モデルのアンサンブル（多数決/平均化）**
# 1. 可視化とエラー解析
#     - **混同行列や確率分布図・ヒートマップでどの被験者/セットで間違えやすいか探索**
#     - **特徴量の重要度解析（Permutation Importanceなど）**
# 1. ハイパーパラメータ探索
#     - **OptunaやGridSearchCV, RandomizedSearchCVによるモデルパラメータ自動最適化**
