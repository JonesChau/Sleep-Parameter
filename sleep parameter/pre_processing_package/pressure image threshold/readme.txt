選取合適的門檻值

方法一. OTSU
在計算機視覺和影像處理中，OTSU用來自動對基於聚類的影像進行二值化，或者說，將一個灰階影像退化為二階影像，該算法是日本學者大津展之提出的。該演算法假定該影像根據雙模直方圖（前景影像與背景影像）把包含兩個灰階值，於是它要計算能將兩類分開的最佳門檻值，所得它們的類內方差最小；由於兩兩平方距離恆定，所以即它們的類間方差最大。然而可用OTSU處理的影像直方圖的分佈為雙峰性，不適用於本論文的壓力影像直方圖。

方法二. triangle threshold
最適用於單個波峰，最開始用於醫學分割細胞等
原理：
1.圖像轉灰度
2.計算圖像灰度直方圖
3.尋找直方圖中兩側邊界
4.尋找直方圖最大值
5.檢測是否最大波峰在亮的一側，否則翻轉
6.計算閾值得到閾值T,如果翻轉則255-T
方法三：Balanced histogram thresholding
This method weighs the histogram, checks which of the two sides is heavier, and removes weight from the heavier side until it becomes the lighter. It repeats the same operation until the edges of the weighing scale meet.
Given its simplicity, this method is a good choice as a first approach when presenting the subject of automatic image thresholding.

方法四：histogram equlization（直方圖等化）
直方圖等化是影像處理領域中利用影像直方圖對對比度進行調整的方法。
这种方法通常用来增加许多影像的全局对比度，尤其是当影像的有用数据的对比度相当接近的时候。通过这种方法，亮度可以更好地在直方图上分布。这样就可以用于增强局部的对比度而不影响整体的对比度，直方圖等化通过有效地扩展常用的亮度来实现这种功能。
这种方法对于背景和前景都太亮或者太暗的图像非常有用，这种方法尤其是可以带来X光图像中更好的骨骼结构显示以及曝光过度或者曝光不足照片中更好的细节。这种方法的一个主要优势是它是一个相当直观的技术并且是可逆操作，如果已知等化函数，那么就可以恢复原始的直方图，并且计算量也不大。这种方法的一个缺点是它对处理的数据不加选择，它可能会增加背景噪声的对比度并且降低有用信号的对比度。

方法五：壓力影像權重比例
原理：
1.將影像轉直方圖
2.尋找直方圖中兩側邊界
3.然後從壓力值大的邊界開始往前面掃，壓力值依次疊加
4.再計算疊加壓力值與壓力值總和的佔比，即比例。
5.比例從90%到99%，再選擇合適的比例，沒超過該比例的當前壓力值即為門檻值

最後我們選擇了方法五


1.壓力影像（Bedsheet Sensor Raw Pressure Map）20 * 11 = 220 
2.前處理 Part 1（Pre-Processing Part1）:定門檻值(Threshold)，將低於門檻值的壓力值變為0，等於或大於的則保持其原來的壓力值，則得到新的壓力影像（New Pressure Image），本論文定的門檻值設定為22。