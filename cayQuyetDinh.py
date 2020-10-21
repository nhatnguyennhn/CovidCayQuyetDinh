import numpy 
import pandas 
from sklearn.tree import DecisionTreeClassifier
duLieuTrain = pandas.read_csv("train.csv", index_col = 'Id')
duLieuTest = pandas.read_csv("test.csv", index_col = 'ForecastId')
pandas.set_option('display.max_columns', 150)
pandas.set_option('display.max_rows', 150)
yTrainSoTruongHop = numpy.array(duLieuTrain['ConfirmedCases'].astype(int))
yTrainSoNguoiChet = numpy.array(duLieuTrain['Fatalities'].astype(int))
cotSoLieu = ['ConfirmedCases', 'Fatalities']

duLieuChung = pandas.concat([duLieuTrain.drop(cotSoLieu, axis=1), duLieuTest])
chiSo = duLieuTrain.shape[0]
duLieuChung = pandas.get_dummies(duLieuChung, columns=duLieuChung.columns)
xTrain = duLieuChung[:chiSo]
xTest= duLieuChung[chiSo:]
tree_model = DecisionTreeClassifier()
tree_model.fit(xTrain, yTrainSoTruongHop)

yTestSoTruongHop = tree_model.predict(xTest)
yTestSoTruongHop= yTestSoTruongHop.astype(int)
yTestSoTruongHop[yTestSoTruongHop <0]=0
tree_model = DecisionTreeClassifier()
tree_model.fit(xTrain, yTrainSoNguoiChet)
yTestSoNguoiChet = tree_model.predict(xTest)
yTestSoNguoiChet= yTestSoNguoiChet.astype(int)
yTestSoNguoiChet[yTestSoNguoiChet <0]=0

ketQua = pandas.DataFrame([yTestSoTruongHop, yTestSoNguoiChet], index = ['SoTruongHop','SoNguoiChet'], columns= numpy.arange(1,yTestSoTruongHop.shape[0] + 1)).T
ketQua.to_csv('ketqua.csv', index_label = "ID")
