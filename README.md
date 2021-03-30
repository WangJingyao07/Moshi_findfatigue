# Moshi_findfatigue
#_author_Wang Jingyao
#模式识别大作业：疲劳检测（检测司机是否有疲劳（或其他有害与行车）的症状或行为）

'''
我们要做的就是检测照片中驾驶员的姿态，共有十个不同的姿态，比如安全驾驶、发短信、喝水、整理礼仪和打电话等。我们可以把这个问题看作一个图片分类的问题，
数据集选自kaggle2015年疲劳检测比赛，共包含4.02GB张图片，测试用例（.csv），以及提交样例：
提供了22424张训练集和79729张测试集，每张照片都是640x480x3，共有10个分类。
本代码共包含三种深度学习架构（然鹅只成功了一个）：pytorch,tensorflow,keras.
一共根据论文对三种四种方法进行了复现：

法一：
“pre-trained Vgg16, and Vgg16_3, a modified version of Vgg16 I devised to deal with overfit.”
VGG16
法二：
融合了4个ResNet152和VGG16模型
在预测样本中，很多图片都是连续帧的，利用当前图片和近邻图片来测试
法三：
CNN
法四:
加了一个融合VGG16

最终利用visdom进行可视化。
'''
