{ [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import yfinance as yf\n",
    "import tensorflow as tf"
   
    "start = '2012-01-01'\n",
    "end = '2022-12-21'\n",
    "stock = 'GOOG'\n",
    "\n",
    "data = yf.download(stock, start, end)"
   
      "text/plain": [
       "                 Open       High        Low      Close  Adj Close     Volume\n",
       "Date                                                                        \n",
       "2012-01-03  16.262545  16.641375  16.248346  16.573130  16.573130  147611217\n",
       "2012-01-04  16.563665  16.693678  16.453827  16.644611  16.644611  114989399\n",
       "2012-01-05  16.491436  16.537264  16.344486  16.413727  16.413727  131808205\n",
       "2012-01-06  16.417213  16.438385  16.184088  16.189817  16.189817  108119746\n",
       "2012-01-09  16.102144  16.114599  15.472754  15.503389  15.503389  233776981\n",
       "...               ...        ...        ...        ...        ...        ...\n",
       "2022-12-14  95.540001  97.220001  93.940002  95.309998  95.309998   26452900\n",
       "2022-12-15  93.540001  94.029999  90.430000  91.199997  91.199997   28298800\n",
       "2022-12-16  91.199997  91.750000  90.010002  90.860001  90.860001   48485500\n",
       "2022-12-19  90.879997  91.199997  88.925003  89.150002  89.150002   23020500\n",
       "2022-12-20  88.730003  89.779999  88.040001  89.629997  89.629997   21976800\n",
       "\n",
       "[2761 rows x 6 columns]"
      
    "data"
   
      "text/plain": [
       "           Date       Open       High        Low      Close  Adj Close  \\\n",
       "0    2012-01-03  16.262545  16.641375  16.248346  16.573130  16.573130   \n",
       "1    2012-01-04  16.563665  16.693678  16.453827  16.644611  16.644611   \n",
       "2    2012-01-05  16.491436  16.537264  16.344486  16.413727  16.413727   \n",
       "3    2012-01-06  16.417213  16.438385  16.184088  16.189817  16.189817   \n",
       "4    2012-01-09  16.102144  16.114599  15.472754  15.503389  15.503389   \n",
       "...         ...        ...        ...        ...        ...        ...   \n",
       "2756 2022-12-14  95.540001  97.220001  93.940002  95.309998  95.309998   \n",
       "2757 2022-12-15  93.540001  94.029999  90.430000  91.199997  91.199997   \n",
       "2758 2022-12-16  91.199997  91.750000  90.010002  90.860001  90.860001   \n",
       "2759 2022-12-19  90.879997  91.199997  88.925003  89.150002  89.150002   \n",
       "2760 2022-12-20  88.730003  89.779999  88.040001  89.629997  89.629997   \n",
       "\n",
       "         Volume  \n",
       "0     147611217  \n",
       "1     114989399  \n",
       "2     131808205  \n",
       "3     108119746  \n",
       "4     233776981  \n",
       "...         ...  \n",
       "2756   26452900  \n",
       "2757   28298800  \n",
       "2758   48485500  \n",
       "2759   23020500  \n",
       "2760   21976800  \n",
       "\n",
       "[2761 rows x 7 columns]"
      
   
    
    
    "data.reset_index(inplace=True)\n",
    "data"
   
    "ma_100_days = data.Close.rolling(100).mean()"
   
     "data": {
      "image/png": "
       "<Figure size 800x600 with 1 Axes>"
      
    "plt.figure(figsize=(8,6))\n",
    "plt.plot(ma_100_days,'r')\n",
    "plt.plot(data.Close,'g')\n",
    "plt.show()"
   
    "ma_200_days = data.Close.rolling(200).mean()"
   
     "data": {
      "image/png": "",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      
    "plt.figure(figsize=(8,6))\n",
    "plt.plot(ma_100_days,'r')\n",
    "plt.plot(ma_200_days,'b')\n",
    "plt.plot(data.Close,'g')\n",
    "plt.show()"
   
    "data.dropna(inplace=True)"
   
    "data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])\n",
    "data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])"
   
     "data": {
      "text/plain": [
       "(2208, 553)"
      
    "data_train.shape[0], data_test.shape[0]"
   
    "from sklearn.preprocessing import MinMaxScaler\n",
    "scaler = MinMaxScaler(feature_range = (0,1))\n",
    "\n",
    "data_train_scale = scaler.fit_transform(data_train)\n"
   
    "x =[]\n",
    "y =[]\n",
    "\n",
    "for i in range(100, data_train_scale.shape[0]):\n",
    "    x.append(data_train_scale[i-100:i])\n",
    "    y.append(data_train_scale[i,0])"
   
    "x, y  = np.array(x), np.array(y)"
   
    "import tensorflow as tf \n",
    "from tensorflow.keras.layers import Dropout, Dense, LSTM\n",
    "from tensorflow.keras.models import Sequential"
   
    "model = Sequential()\n",
    "model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,\n",
    "               input_shape = ((x.shape[1],1))))\n",
    "model.add(Dropout(0.2))\n",
    "\n",
    "model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))\n",
    "model.add(Dropout(0.3))\n",
    "\n",
    "model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))\n",
    "model.add(Dropout(0.4))\n",
    "\n",
    "model.add(LSTM(units = 120, activation = 'relu'))\n",
    "model.add(Dropout(0.5))\n",
    "\n",
    "model.add(Dense(units =1))\n",
    "\n"
   
    "model.compile(optimizer = 'adam', loss = 'mean_squared_error')"
   
    "model.fit(x,y, epochs = 50, batch_size = 32, verbose = 1)"
   
    "model.summary()"
   
    "past_100_days = data_train.tail(100)"
   
    "data_test = pd.concat([past_100_days, data_test], ignore_index = True)\n",
    "\n",
    "data_test"
   
    "data_test_scale = scaler.fit_transform(data_test)"
   
    "x =[]\n",
    "y =[]\n",
    "\n",
    "for i in range(100, data_test_scale.shape[0]):\n",
    "    x.append(data_test_scale[i-100:i])\n",
    "    y.append(data_test_scale[i,0])\n",
    "    \n",
    "x, y  = np.array(x), np.array(y)"
   
    "y_predict = model.predict(0)\n",
    "y_predict"
   
    "scaler.scale_"
   
    "scale = 1/scaler.scale_\n",
    "scale"
   
    "y_predict = y_predict*scale"
   
    "y = y*scale"
   
    "plt.figure(figsize = (10,8))\n",
    "plt.plot(y_predict, 'r', label = 'Predicted Price')\n",
    "plt.plot(y, 'g', label = 'Original Price')\n",
    "plt.xlabel('time')\n",
    "plt.ylabel('Price')\n",
    "plt.legend()\n",
    "plt.show()"
   
    "model.save('Stock Price Prediction Model.keras')"
   ]}




          
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
