# TTI Numerical Calculation course report (2020/06/30)  18037 Yuki. Kondo
# -*- coding: utf-8 -*-

"""
=================================================================
PCA解析プログラム(CPUスペック解析用)
スクリプト実行 （レポート小問3の回答を出力)
PCAオブジェクトを生成し，mainメソッドの実行により，各解析メソッドが実行される．

(統計学の前提条件)：不偏分散を基準とする
(注意)グラフにはメイリオを使用しており，Linux,Unix環境ではインストールされていない可能性がある．
=================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# SVDクラス
class SVD:
    # 行列Aの定義と出力
    def __init__(self):
        # ----小問1用の初期設定 オーバーライドで無効化----
        self.A = np.array([[7, 3], [3, 13], [13, 7]])
        divPrint("matrix A", self.A)

    # svdの実施(Wを対角行列に変換)
    def calc(self):
        U, W, Vt = np.linalg.svd(self.A, full_matrices=False)
        W = np.diag(W)
        return U, W, Vt

    # svdによって分解された行列の表示
    def print_UWVt(self):
        U, W, Vt = self.calc()
        divPrint("matrix U", U, "matrix W", W, "matrix V", Vt.T)

    # 行列AとUWVtの比較
    def validate(self):
        U, W, Vt = self.calc()
        a_calc = U @ W @ Vt
        divPrint("行列積 UWVt", a_calc)
        if np.array_equal(self.A, a_calc):
            compare = "行列が完全一致しました。"  # 比較する行列要素a,bがa==b をすべての要素において満たす場合,True
        elif np.allclose(self.A, a_calc):
            compare = "行列が誤差範囲内で一致しました。\n*比較する行列要素a,bがabs(a - b）<=（1e-8 + 1e-5 *abs(b)）\nをすべての要素において満たしている。"  #
            # 比較する行列要素a,bがabs(a - b）<=（1e-8 + 1e-5 *abs(b)） をすべての要素において満たす場合,True
        else:
            compare = "行列は一致しません。"  # 行列が誤差範囲内にも一致しない場合
        divPrint("UWVtとAの比較", compare, "**行列間の誤差(A-UWVt)**", self.A - a_calc)  # 結果を出力


# グラフ描画クラス
class Graph:
    # 初期設定メソッド
    def __init__(self):
        # ----ユーザー設定値----
        self.count = 1  # 保存画像のナンバリング初期値
        self.fontsize = "small"  # 散布図のテキストのフォントサイズの設定   x0.833 相対的なサイズ
        self.lim = [-12, 12]  # 散布図グラフ軸の[最小値,最大値] deff[-12, 12]
        self.smallLim = [-1, 1]  # 縮小版のグラフ範囲    deff [1, -1]
        self.saveDirectory = "result/"  # 保存するディレクトリを指定
        if not os.path.exists(self.saveDirectory):  # ディレクトリが存在しない場合
            os.makedirs(self.saveDirectory)  # ディレクトリを作成
        plt.rcParams['font.family'] = 'meiryo'  # グラフに使用するフォント設定。(Linux, Unix環境ではmeiryoを要インストール)
        self.labelOffset = [240, 330]  # x,y軸ラベルオフセットの自動調整用変数[x, y]    deff [240, 330]
        self.vectorMagnification = 12  # ベクトル倍率    deff 12

    # 散布図作成・保存メソッド
    def scatterDraw(self, x, y, dataLabels, scatterLabels=('scatter plot', "x", "y"), x2=None, y2=None,
                    dataLabels2=None, smallFlag=False, figName=""):  # x:プロットxデータ,y:プロットyデータ,
        # dataLabels:テキストのユニークなデータラベル,scatterLabels:グラフタイトル,x軸タイトル、y軸タイトル, x2,y2:第二データ,dataLabels2:第二データ
        # テキストのユニークなデータラベル, smallFlag:グラフ軸範囲の縮小, figName：参照メソッドに関連する名前,

        # ---グラフ設定----
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))  # 拡張性確保のためにsubplotで生成．16*12インチで設定A
        plt.subplots_adjust(bottom=0.15)  # グラフの下部空白を拡張
        # x,y軸を原点基準に移動
        ax.spines['bottom'].set_position("zero")
        ax.spines['left'].set_position("zero")

        # プロット点のカラーマップの設定
        color_value, cmap = self.scatterPlotColorSelect(dataLabels)

        # グラフのx,y軸の最大・最小値を設定
        if smallFlag:  # 縮小版のグラフ範囲を設定
            ax.set_xlim(self.smallLim)  # x軸のグラフ範囲を設定
            ax.set_ylim(self.smallLim)  # y軸のグラフ範囲を設定
            yTextOffset = 0.05  # テキストのy軸方向オフセット
        else:  # 通常のグラフ範囲を設定
            ax.set_xlim(self.lim)  # x軸のグラフ範囲を設定
            ax.set_ylim(self.lim)  # y軸のグラフ範囲を設定
            yTextOffset = 0.2  # テキストのy軸方向オフセット deff 0.3

        # ----グラフ作成-----
        ax.scatter(x, y, edgecolors="black", c=color_value, cmap=cmap)  # 散布図を作成、カラーマップ　gist_ncar or bwr
        ax.set_title(scatterLabels[0], y=-.2, )  # タイトル設定、枠外下部に来るように調整
        ax.set_xlabel(scatterLabels[1], labelpad=self.labelOffset[0])  # x軸ラベルの設定、枠外下部に来るように調整
        ax.set_ylabel(scatterLabels[2], labelpad=self.labelOffset[1])  # y軸ラベルの設定、枠外左部に来るように調整
        plt.grid()  # グリッドを追加
        for i in range(len(x)):  # データ全点にラベリング
            plt.text(x[i], y[i] + yTextOffset, dataLabels[i], ha='center', va='center', fontsize=self.fontsize)  #
            # 見やすい位置にオフセット

        # ---第2成分が存在するときの追加処理（因子負荷率をベクトル化して、主成分得点を主成分とパラメータで視覚的に評価する）---
        if x2 is not None and y2 is not None and dataLabels2 is not None:
            for i in range(len(x2)):
                end = (x2[i] * self.vectorMagnification, y2[i] * self.vectorMagnification)  # ベクトルの大きさを設定倍する
                ax.annotate('', xy=(0, 0), xytext=end,  # ベクトルを追加
                            arrowprops=dict(arrowstyle='<-',  # 矢印形状
                                            connectionstyle='arc3',  # 直線のベクトル
                                            facecolor='#00FFFF',  # 矢印の色を設定
                                            edgecolor='#00FFFF')  # 矢印の枠の色を設定
                            )
                plt.text(end[0] + 0.2, end[1] + 0.2, dataLabels2[i], ha='center', va='center', fontsize=self.fontsize,
                         color="blue")  # 青色のテキストでベクトルのパラメータ名を表示
            # plt.show()
        # plt.show()

        # ---グラフの保存---
        fig.savefig(f"{self.saveDirectory}scatter{self.count}_{figName}.png")  # 画像の保存
        plt.close()  # 大量画像データ処理によるメモリ不足エラーを避ける．
        self.count += 1  # 保存用ナンバーの更新

    # パレート図作成・保存メソッド
    def paretoChartDraw(self, y1, y2, graphLabels=('lineGraph', "x", "y1", "y2")):  # x:プロットxデータ,y:プロットyデータ,
        # graphLabels:グラフタイトル,x軸タイトル、y軸タイトル
        data_num = len(y1)
        # ---グラフ設定----
        fig, ax1 = plt.subplots(figsize=(6,4))  # 拡張性確保のためにsubplotで生成．デフォルト値 8*6インチで設定
        fig.subplots_adjust(right=0.85)
        accum_to_plot = [0] + list(y2)
        colorcodes = ['#C4F092', '#EEFA98', '#E3DE96', '#FAEA98', '#F0D792' ,'#F4D002' ,'#FFFF51',
                    '#80C686' ,'#509458' ,'#1F662E' ,'#1168AD' ,'#003E7D' ,'#001950']
        percent_labels = [str(i) + "%" for i in np.arange(0, 100+1, 10)]

        ax1.bar(range(1, data_num + 1), y1, align="edge", width=-1, edgecolor='k', color=colorcodes[:data_num+1])

        ax1.set_xticks([0.5 + i for i in range(data_num)], minor=True)
        ax1.set_xticklabels([i+1 for i in range(data_num)], minor=True)
        ax1.tick_params(axis="x", which="major", direction="in")
        ax1.set_ylim([0, sum(y1)])
        ax1.set_xlabel(graphLabels[1])
        ax1.set_ylabel(graphLabels[2])

        ax2 = ax1.twinx()
        ax2.set_xticks(range(data_num+1))
        ax2.plot(range(data_num+1), accum_to_plot, color="red", marker="o")
        ax2.set_xticklabels([])
        ax2.set_xlim([0,data_num])
        ax2.set_ylim([0, 100])
        ax2.set_yticks(np.arange(0, 100+1, 10))
        ax2.set_yticklabels(percent_labels)
        ax2.set_ylabel(graphLabels[3])
        # ax2.grid(True)

        x = range(data_num)
        for i, j in zip(x, accum_to_plot):
            plt.annotate(f"{j:.2f}", xy=(i+0.1, j-5),color = "red")
        for i, j in zip(x, y1):
            plt.annotate(f"{j:.2f}", xy=(i+0.1, 9*j+1),color = "black")

        # ---グラフ作成---
        fig.savefig(f"{self.saveDirectory}lineGraph{self.count}.png")  # 画像の保存
        plt.close()  # メモリ不足エラーを避ける．
        # plt.show()

    # 散布図カラーマップ設定メソッド
    def scatterPlotColorSelect(self, dataLabels):
        color_value = np.random.randint(30, 80, len(dataLabels))  # プロットするデータの色を決める乱数リストを作成
        cmap = "gist_ncar"  # カラーマップをgist_ncarに設定
        return color_value, cmap  # 二つの変数を返す


# SVD,Graphクラスを継承
class PCA(SVD, Graph):
    # 初期設定メソッド
    def __init__(self, dataName):
        Graph.__init__(self)  # Graphクラスの初期設定メソッドを実行
        self.df = pd.read_csv(f"./{dataName}", encoding="shift-jis", index_col=0)  # データを抽出
        self.header = list(self.df.columns)  # パラメータ名リスト
        self.index = list(self.df.index)  # ユニークなサンプル名リスト
        divPrint("サンプル名", self.index, "パラメータ", self.header)  # サンプル名、パラメータ名を出力
        self.source = self.df.values  # データフレームをndarrayに変換
        self.A = np.zeros(self.source.shape)  # 標準化したデータの行列を格納する
        self.numPrincipalComponents = None  # 主成分数の数(Rank落ちを考慮)
        self.principalComponentScoreMatrix = None  # 主成分得点行列 (行:サンプル、列：主成分 ).(numPrincipalComponentsが定まるまで不定)
        self.factorLoadingMatrix = None  # 因子負荷量行列 (行:パラメータ、列：主成分 ).(numPrincipalComponentsが定まるまで不定)
        self.cumulativeContributionRatio = None  # 組み合わせを格納する(combinationFunctionで使用)
        divPrint("行列化したデータ", self.source)  # 行列化したデータを表示
        self.splitIntelAMD = False  # AMDとIntelの色分けフラグ(オーバーライドしたscatterPlotColorSelectで利用)
        # ----ユーザー設定値----
        self.numCutScatter = 100  # 散布図を作成する主成分の上限を設定．ex:第3主成分の組み合わせまで->3を入力　使用しない場合，十分に大きな値を入力
        self.seriesAMD = ["Athlon", "Ryzen"]  # AMD製CPUのシリーズ名称リスト(散布図の色分けのためのリスト)

    # 標準化メソッド
    def standardization(self):
        for i in range(0, self.source.shape[1]):  # パラメータごとに標準化後、行列Aに格納。forは列数だけループ
            row_data = self.source[:, i]  # 列データをベクトルとして抽出
            mean = np.mean(row_data)  # 列データの平均値を算出
            std = np.std(row_data, ddof=1)  # 列データの標準偏差を算出.不偏分散に基づくのでddof=1
            self.A[:, i] = (row_data - mean) / std  # ベクトル要素全てに標準化を実施し、同じ列に格納
        divPrint("標準化後のデータ行列", self.A)

    # SVD U,W,VT行列取得・出力・比較メソッド
    def getSvdMatrix(self):
        self.U, self.W, self.Vt = self.calc()  # SVDクラスのcalcメソッドを実施し、行列を取得
        self.Wvector = np.diag(self.W)  # W行列をベクトルとしても保存
        self.numPrincipalComponents = np.count_nonzero(self.Wvector != 0)  # 特異値が0でない要素数=主成分数
        self.principalComponentScoreMatrix = np.zeros((self.A.shape[0], self.numPrincipalComponents))  # 主成分得点行列
        # (行:サンプル、列：主成分 )の初期化
        self.factorLoadingMatrix = np.zeros((self.A.shape[1], self.numPrincipalComponents))  # 因子負荷量行列 (行:パラメータ、列：主成分)
        # の初期化
        self.print_UWVt()  # 特異値分解結果を出力
        self.validate()  # SVDクラスのcalcメソッドを実施し、行列を比較
        divPrint(f"主成分数:{self.numPrincipalComponents}成分")

    # (累積)寄与率計算・出力メソッド
    def contributionRatioFunction(self):
        self.contributionRatio = self.Wvector ** 2 / (self.A.shape[0]-1)  # Wベクトルの各要素を二乗し、その値をAの行数すなわちサンプル数-1で割ることで寄与率を
        # 1次元配列として取得
        self.cumulativeContributionRatio = np.cumsum(self.contributionRatio)  # 上記の配列を順次累積していった累積寄与率の1次元配列を取得
        # ---寄与率、累積寄与率の出力---
        for i in range(self.numPrincipalComponents):  # 主成分数だけループで処理
            divPrint(f"**第{i + 1}主成分**",
                     f"寄与率: {self.contributionRatio[i]:.6f}",
                     f"寄与率(%換算): {self.contributionRatio[i] / np.sum(self.contributionRatio) * 100:.6f} %",
                     f"累積寄与率: {self.cumulativeContributionRatio[i]:.6f}",
                     f"累積寄与率(%換算): {self.cumulativeContributionRatio[i] / np.sum(self.contributionRatio) * 100:.6f} %",
                     )

    # 主成分得点計算・出力メソッド
    def principalComponentScoreFunction(self):
        for i in range(self.numPrincipalComponents):  # 主成分の数だけループ
            principalComponentScore = np.dot(self.A, self.Vt[i, :])  # 標準化されたデータの行列とVtの行(Vの列)ベクトルの積を計算
            self.principalComponentScoreMatrix[:, i] = principalComponentScore  # 行：サンプル、列:主成分として行列に上記の1次元配列として対象列に格納
            divPrint(f"**第{i + 1}主成分得点**", "サンプル名 : 値",
                     *[f"{self.index[j]} : {principalComponentScore[j]}" for j in
                       range(len(principalComponentScore))])  # 対象の主成分の主成分得点を表示

    # 因子負荷量計算・出力メソッド
    def factorLoadingFunction(self):
        for i in range(self.numPrincipalComponents):  # 主成分の数だけループ
            principalComponentScore = self.principalComponentScoreMatrix[:, i]  # 主成分得点行列から対象の主成分の主成分得点の1次元配列を取得
            for j in range(0, self.A.shape[1]):  # パラメータ数(データ行列の列数)だけ
                row_data = self.A[:, j]  # データ行列の列ベクトルを取得
                covariance = np.cov(principalComponentScore, row_data, bias=False)  # 主成分得点と上記の列ベクトルから共分散行列を生成.(非対角：共分散、対角：それぞれの分散)
                factorLoading = covariance[0, 1] / ((covariance[0, 0] ** 0.5) * (covariance[1, 1] ** 0.5))  # 因子負荷量を計算
                self.factorLoadingMatrix[j][i] = factorLoading  # 因子負荷量行列(行：パラメータ、列：主成分)に因子負荷量を入れる
            divPrint(f"**第{i + 1}主成分 因子負荷量**", "パラメータ名 : 値",
                     *[f"{self.header[j]} : {self.factorLoadingMatrix[j][i]}" for j in
                       range(self.A.shape[1])])  # 対象の主成分の因子負荷量を表示

    # 主成分得点ー累積寄与率　棒グラフ出力メソッド
    def graphCumulativeContributionRatio(self):
        graphLabels = ("主成分数と累積寄与率の関係", "主成分", "寄与率", "累積寄与率[%]")  # 左からグラフタイトル, x軸ラベル, y軸ラベル
        y1 = self.contributionRatio  # 因子負荷率の1次元配列。
        y2 = self.cumulativeContributionRatio / self.cumulativeContributionRatio[-1] * 100
        self.paretoChartDraw(y1, y2, graphLabels=graphLabels)  # GraphクラスのlineGraphDrawメソッドを実行

    # 因子負荷量 散布図出力メソッド
    def sFactorLoading(self):
        conb = self.combinationFunction(self.numPrincipalComponents)  # 主成分の組み合わせの2次元配列を取得
        for i in range(conb.shape[1]):  # 組み合わせ数だけループ
            if self.numCutScatter <= conb[1][i] or self.numCutScatter <= conb[0][i]:  # 主成分数が上限値以上であった場合，グラフ生成を中止
                continue
            scatterLabels = ("各パラメータに関する因子負荷量の関係性", f"第{conb[1][i] + 1}主成分", f"第{conb[0][i] + 1}主成分")  # 左からグラフタイトル,
            # x軸ラベル, y軸ラベル
            self.scatterDraw(self.factorLoadingMatrix[:, conb[1][i]],  # x軸の因子負荷量配列（組み合わせ配列の要素をインデックスとして引く）
                             self.factorLoadingMatrix[:, conb[0][i]],  # y軸の因子負荷量配列（組み合わせ配列の要素をインデックスとして引く）
                             self.header,  # テキストラベル用のパラメータ名配列
                             scatterLabels=scatterLabels,  # グラフタイトル, x軸ラベル, y軸ラベルの情報
                             smallFlag=True,  # 因子負荷量 散布図用のグラフ軸の領域のフラグ
                             figName="FactorLoading")  # 参照メソッドに関連す保存名

    # 主成分得点 散布図出力メソッド
    def sPrincipalComponentScore(self):
        conb = self.combinationFunction(self.numPrincipalComponents)  # 主成分の組み合わせの2次元配列を取得
        for i in range(conb.shape[1]):  # 組み合わせ数だけループ
            if self.numCutScatter <= conb[1][i] or self.numCutScatter <= conb[0][i]:  # 主成分数が上限値以上であった場合，グラフ生成を中止
                continue
            scatterLabels = ("各サンプルに関する主成分得点の関係性", f"第{conb[1][i] + 1}主成分", f"第{conb[0][i] + 1}主成分")  # 左からグラフタイトル,
            # x軸ラベル, y軸ラベル
            self.scatterDraw(self.principalComponentScoreMatrix[:, conb[1][i]],  # x軸の主成分得点配列（組み合わせ配列の要素をインデックスとして引く）
                             self.principalComponentScoreMatrix[:, conb[0][i]],  # y軸の主成分得点配列（組み合わせ配列の要素をインデックスとして引く）
                             self.index,  # テキストラベル用のサンプル名配列
                             scatterLabels=scatterLabels,  # グラフタイトル, x軸ラベル, y軸ラベルの情報
                             figName="PrincipalComponentScore")  # 参照メソッドに関連する保存名

    # 因子負荷量をベクトルとした主成分得点 散布図出力メソッド
    def sFLandPCS(self, splitIntelAMD=False):
        self.splitIntelAMD = splitIntelAMD  # 色分けフラグを立てる
        conb = self.combinationFunction(self.numPrincipalComponents)  # 主成分の組み合わせの2次元配列を取得
        for i in range(conb.shape[1]):  # 組み合わせ数だけループ
            if self.numCutScatter <= conb[1][i] or self.numCutScatter <= conb[0][i]:  # 主成分数が上限値以上であった場合，グラフ生成を中止
                continue
            scatterLabels = ("主成分軸上の各パラメータと各サンプルの関係", f"第{conb[1][i] + 1}主成分", f"第{conb[0][i] + 1}主成分")  #
            # 左からグラフタイトル, x軸ラベル, y軸ラベル
            if splitIntelAMD:
                figName = "FLandPCS_splitIntelAMD"  # 参照メソッドに関連する保存名(Intel AMDの色分けあり)
            else:
                figName = "FLandPCS"  # 参照メソッドに関連する保存名
            self.scatterDraw(self.principalComponentScoreMatrix[:, conb[1][i]],  # x軸の主成分得点配列（組み合わせ配列の要素をインデックスとして引く）
                             self.principalComponentScoreMatrix[:, conb[0][i]],  # y軸の主成分得点配列（組み合わせ配列の要素をインデックスとして引く）
                             self.index,  # テキストラベル用のサンプル名配列
                             scatterLabels=scatterLabels,  # グラフタイトル, x軸ラベル, y軸ラベルの情報
                             x2=self.factorLoadingMatrix[:, conb[1][i]],  # x軸の因子負荷量配列（組み合わせ配列の要素をインデックスとして引く）
                             y2=self.factorLoadingMatrix[:, conb[0][i]],  # y軸の因子負荷量配列（組み合わせ配列の要素をインデックスとして引く）
                             dataLabels2=self.header,  # テキストラベル用のパラメータ名配列
                             figName=figName)  # 参照メソッドに関連する保存名
        self.splitIntelAMD = False  # 色分けフラグをリセットする

    # 組み合わせ生成メソッド
    def combinationFunction(self, element):
        element = np.arange(element)  # 要素を1次元配列とする
        xx, yy = np.meshgrid(element, element)  # 要素の1次元配列をそれぞれ横、縦にコピーして並べた2次元配列を取得
        xx = xx[np.triu_indices(len(element), k=1)]  # 対角成分を覗いた上三角行列成分をベクトル化
        yy = yy[np.triu_indices(len(element), k=1)]  # 対角成分を覗いた上三角行列成分をベクトル化
        conb = np.zeros((2, len(xx)))  # 組み合わせ配列。列ごとに組み合わせを格納。nC2（行数：2,列数：組み合わせ数）
        conb[0, :] = xx  # 1列目に格納
        conb[1, :] = yy  # 2列目に格納
        # print(conb)
        return conb.astype("int64")  # ラベル表示のために整数化して返す

    # =========================上位クラスのメソッドのオーバーライド======================================
    # 散布図カラーマップ設定メソッド
    def scatterPlotColorSelect(self, dataLabels):
        if self.splitIntelAMD:  # Intel AMD別に色分けする設定
            color_value = []  # 初期化
            for label in dataLabels:
                for i in range(len(self.seriesAMD)):
                    if self.seriesAMD[i] in label:  # ラベル名がAMDシリーズの場合
                        color_value.append(100)  # 赤色の100を追加
                        break
                    elif i == len(self.seriesAMD) - 1:  # AMDシリーズ名のいずれにも一致しない場合
                        color_value.append(0)  # 青色の0を追加
                cmap = "bwr"  # カラーマップをbwrに設定  100:red(AMD),0:blue(Intel)
            return color_value, cmap
        else:  # フラグがたっていない場合，上位クラスのメソッドを実行(プロットの色をランダムに決める)
            color_value, cmap = Graph.scatterPlotColorSelect(self, dataLabels)  # GraphクラスのscatterPlotColorSelectメソッドを実行
            return color_value, cmap

    # =========================メインメソッド======================================
    def main(self):
        self.standardization()  # 標準化メソッドを実行
        self.getSvdMatrix()  # SVD U,W,VT行列取得・出力・比較メソッドを実行
        self.contributionRatioFunction()  # (累積)寄与率計算・出力メソッドを実行
        self.principalComponentScoreFunction()  # 主成分得点計算・出力メソッドを実行
        self.factorLoadingFunction()  # 因子負荷量計算・出力メソッドを実行
        self.sFactorLoading()  # 因子負荷量 散布図出力メソッドを実行
        self.sPrincipalComponentScore()  # 主成分得点 散布図出力メソッドを実行
        self.graphCumulativeContributionRatio()  # 主成分得点ー累積寄与率　棒グラフ出力メソッドを実行
        self.sFLandPCS()  # 因子負荷量をベクトルとした主成分得点 散布図出力メソッドを実行
        self.sFLandPCS(splitIntelAMD=True)  # 因子負荷量をベクトルとした主成分得点 散布図出力メソッドを実行(AMDとIntelの色分け)


# オリジナルプリント関数(分割表示関数)
def divPrint(*args):  # 任意の数の変数を開業しながら表示。開始と終了を線で区切る。
    print("===========================================================\n")
    for i in args:
        print(f"{i}")
    print("\n===========================================================\n")


# スクリプト実行のときのみ実行。
if __name__ == '__main__':
    pca = PCA("CPU.csv")  # 読み込むcsvを引数で指定したPCAオブジェクトを生成する．
    pca.main()  # PCAクラスのmainメソッドを実行する．
    # print(np.__version__)
    # print(pd.__version__)
